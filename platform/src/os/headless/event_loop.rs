use super::raster::encode_png_rgba;
use crate::{
    cx::Cx,
    cx_api::{CxOsApi, CxOsOp, OpenUrlInPlace},
    event::{Event, TextClipboardEvent, WindowGeom, WindowGeomChangeEvent},
    makepad_live_id::*,
    makepad_math::dvec2,
    makepad_micro_serde::*,
    os::shared_framebuf::{PollTimer, PresentableDraw, PresentableImageId},
    thread::SignalToUI,
    window::CxWindowPool,
};
use makepad_studio_protocol::{AppToStudio, ScreenshotResponse, StudioToApp};
use std::{
    cell::RefCell,
    io::{self, BufRead, BufReader, Write},
    path::PathBuf,
    rc::Rc,
    thread,
    time::Duration,
    time::Instant,
};

/// Backing-store scale for a headless window. Retina by default, because a
/// screenshot is expected to match what a real display would show. Rendering is
/// a software rasteriser here, so the cost is per PIXEL: a suite that only
/// asserts on logical geometry can set `MAKEPAD_HEADLESS_DPI=1` and do a
/// quarter of the work.
fn configured_headless_dpi() -> f64 {
    std::env::var("MAKEPAD_HEADLESS_DPI")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|dpi| *dpi > 0.0)
        .unwrap_or(2.0)
}

#[derive(Default)]
struct HeadlessWindowState {
    created: bool,
    width: u32,
    height: u32,
    dpi_factor: f64,
    frame_id: u64,
    presentable_id: Option<PresentableImageId>,
}

impl HeadlessWindowState {
    fn ensure_size_defaults(&mut self) {
        if self.width <= 1 {
            self.width = 1280;
        }
        if self.height <= 1 {
            self.height = 720;
        }
        if self.dpi_factor <= 0.0 {
            self.dpi_factor = 1.0;
        }
    }
}

/// Tamaño lógico forzado de ventana, leído de `MAKEPAD_HEADLESS_SIZE`
/// (formato `ANCHOxALTO`, p. ej. `1280x720`). `None` si no está definida o no
/// se puede parsear. Cacheado por el mismo motivo que la de abajo.
fn headless_forced_window_size() -> Option<crate::makepad_math::DVec2> {
    use std::sync::OnceLock;
    static SIZE: OnceLock<Option<(f64, f64)>> = OnceLock::new();
    let parsed = *SIZE.get_or_init(|| {
        let raw = std::env::var("MAKEPAD_HEADLESS_SIZE").ok()?;
        let (w, h) = raw.split_once(['x', 'X'])?;
        let w = w.trim().parse::<f64>().ok()?;
        let h = h.trim().parse::<f64>().ok()?;
        if w > 0.0 && h > 0.0 {
            Some((w, h))
        } else {
            None
        }
    });
    parsed.map(|(w, h)| dvec2(w, h))
}

/// ¿Está activo el redibujado forzado por ciclo? (`MAKEPAD_HEADLESS_FORCE_REDRAW`)
///
/// Se consulta una vez y se cachea porque el bucle la mira en cada iteración y
/// `std::env::var` no es gratis (toma un lock global del entorno).
fn headless_force_redraw_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MAKEPAD_HEADLESS_FORCE_REDRAW").is_ok())
}

impl Cx {
    pub fn event_loop(cx: Rc<RefCell<Cx>>) {
        cx.borrow_mut().self_ref = Some(cx.clone());

        if crate::app_main::should_run_stdin_loop_from_env() {
            cx.borrow_mut().in_makepad_studio = true;
            cx.borrow_mut().stdin_event_loop();
        } else {
            let draw_cycles = cx.borrow().os.draw_cycles;
            if let Some(draw_cycles) = draw_cycles {
                cx.borrow_mut().headless_bounded_loop(draw_cycles);
            } else {
                cx.borrow_mut().headless_single_frame();
            }
        }
    }

    pub fn headless_event_loop_for_draw_cycles(cx: Rc<RefCell<Cx>>, draw_cycles: usize) {
        cx.borrow_mut().self_ref = Some(cx.clone());
        cx.borrow_mut().headless_bounded_loop(draw_cycles.max(1));
    }

    pub fn headless_no_draw_event_loop_for_draw_cycles(cx: Rc<RefCell<Cx>>, draw_cycles: usize) {
        cx.borrow_mut().self_ref = Some(cx.clone());
        {
            let mut cx_ref = cx.borrow_mut();
            cx_ref.os.no_draw = true;
            cx_ref.os.no_draw_initialized = false;
        }
        cx.borrow_mut().headless_bounded_loop(draw_cycles.max(1));
    }

    fn headless_single_frame(&mut self) {
        let mut windows = Vec::new();
        self.call_event_handler(&Event::Startup);
        self.headless_handle_platform_ops(&mut windows, false);
        if windows.is_empty() {
            windows.push(HeadlessWindowState {
                created: true,
                width: 1280,
                height: 720,
                dpi_factor: 1.0,
                frame_id: 0,
                presentable_id: None,
            });
        }
        let time_now = self.seconds_since_app_start();
        if !self.new_next_frames.is_empty() {
            self.call_next_frame_event(time_now);
        }
        if self.os.no_draw || self.need_redrawing() {
            let _ = self.headless_process_draw_cycle(&mut windows, false, time_now);
        }
    }

    fn headless_bounded_loop(&mut self, draw_cycles: usize) {
        let mut windows = Vec::new();
        self.call_event_handler(&Event::Startup);
        let mut running = self.headless_handle_platform_ops(&mut windows, false);
        if windows.is_empty() {
            windows.push(HeadlessWindowState {
                created: true,
                width: 1280,
                height: 720,
                dpi_factor: 1.0,
                frame_id: 0,
                presentable_id: None,
            });
        }

        let mut completed_cycles = 0usize;
        while running && completed_cycles < draw_cycles {
            if SignalToUI::check_and_clear_ui_signal() {
                self.handle_termination_signal();
                self.handle_script_signals();
                self.call_event_handler(&Event::Signal);
            }
            if SignalToUI::check_and_clear_action_signal() {
                self.handle_action_receiver();
            }
            self.dispatch_network_runtime_events();

            let timer_events = self.os.stdin_timers.get_dispatch();
            for event in timer_events {
                self.handle_script_timer(&event);
                self.call_event_handler(&Event::Timer(event));
            }

            running = self.headless_handle_platform_ops(&mut windows, false);
            if !running {
                break;
            }

            let time_now = self.os.stdin_timers.time_now();
            if !self.new_next_frames.is_empty() {
                self.call_next_frame_event(time_now);
            }
            // Redibujado forzado (ATLAS/H0, sólo para medir).
            //
            // El bucle headless sólo repinta cuando la UI se ensucia, que es lo
            // correcto para producción pero inútil para cronometrar: una app
            // estática pinta un frame y ya. Con `MAKEPAD_HEADLESS_FORCE_REDRAW`
            // se marca todo sucio en cada ciclo, de modo que `--draws=N`
            // produce N frames completos y se puede separar el coste del PRIMER
            // frame (atlas de glifos frío, cachés vacías) del de régimen.
            if headless_force_redraw_enabled() {
                self.redraw_all();
            }
            if self.os.no_draw || self.need_redrawing() {
                // Reloj de pared del ciclo COMPLETO (evento de draw + render +
                // present). Es la cifra honesta de "cuánto tarda un frame".
                let cycle_start = std::time::Instant::now();
                let _ = self.headless_process_draw_cycle(&mut windows, false, time_now);
                if std::env::var("MAKEPAD_HEADLESS_PROFILE").is_ok() {
                    crate::log!(
                        "[headless][profile] CICLO_TOTAL={:.1}ms",
                        cycle_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
            completed_cycles += 1;

            if !running {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn headless_process_draw_cycle(
        &mut self,
        windows: &mut Vec<HeadlessWindowState>,
        send_protocol: bool,
        time_now: f64,
    ) -> bool {
        if self.os.no_draw {
            self.call_draw_event(time_now);
        crate::studio_tick_watchdog::note_studio_drew();
            crate::studio_tick_watchdog::note_studio_drew();
            self.os.no_draw_initialized = true;
            return false;
        }
        // Instrumentación H0-bis: `call_draw_event` es TODA la parte de UI que no
        // es rasterizado (recorrido del árbol de widgets, layout, shaping de
        // texto, construcción de draw-lists). Va aparte porque el repintado
        // parcial NO la ahorra: es coste por frame, no por píxel.
        let draw_ev_start = std::time::Instant::now();
        self.call_draw_event(time_now);
        if std::env::var("MAKEPAD_HEADLESS_PROFILE").is_ok() {
            crate::log!(
                "[headless][profile] call_draw_event(layout+widgets)={:.1}ms",
                draw_ev_start.elapsed().as_secs_f64() * 1000.0
            );
        }
        // Los dos lados de este conflicto son COMPLEMENTARIOS, no alternativos:
        // arriba el perfilado de ATLAS, aqui el watchdog que trajo el arreglo de
        // "una UI QUIETA dejaba imagenes del swapchain sin estrenar". Ninguno de
        // los dos falla al faltar, asi que quedarse con uno no daria sintoma.
        crate::studio_tick_watchdog::note_studio_drew();
        self.headless_compile_shaders();
        if send_protocol && self.screenshot_requests.is_empty() {
            self.headless_render_all_passes(time_now);
            true
        } else {
            self.headless_emit_frames(windows, send_protocol, time_now)
        }
    }

    pub fn stdin_event_loop(&mut self) {
        Cx::set_studio_stdout_mode(true);
        let (json_msg_tx, json_msg_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(std::io::stdin().lock());
            let mut line = String::new();
            loop {
                line.clear();
                if let Ok(0) | Err(_) = reader.read_line(&mut line) {
                    break;
                }
                match StudioToApp::deserialize_json(&line) {
                    Ok(msg) => {
                        if json_msg_tx.send(msg).is_err() {
                            break;
                        }
                    }
                    Err(err) => {
                        crate::error!("Cant parse stdin-JSON {} {:?}", line, err);
                    }
                }
            }
        });

        let mut windows = Vec::<HeadlessWindowState>::new();
        write_stdout_msg(&AppToStudio::BeforeStartup);
        self.call_event_handler(&Event::Startup);
        let mut running = self.headless_handle_platform_ops(&mut windows, true);
        if running {
            let time_now = self.seconds_since_app_start();
            if self.os.no_draw || self.need_redrawing() {
                let _ = self.headless_process_draw_cycle(&mut windows, true, time_now);
            }
        }
        write_stdout_msg(&AppToStudio::AfterStartup);
        // Nothing below turns a dirty tree into a frame except the Tick
        // branch, so a host that never ticks fails in total silence. The
        // watchdog is the only thing that can notice, because with no ticks
        // this loop is parked waiting and no code of ours runs.
        crate::studio_tick_watchdog::start_studio_tick_watchdog();

        while running {
            let msg = match json_msg_rx.recv() {
                Ok(msg) => msg,
                Err(_) => break,
            };
            match msg {
                StudioToApp::KeyDown(e) => self.call_event_handler(&Event::KeyDown(e)),
                StudioToApp::KeyUp(e) => self.call_event_handler(&Event::KeyUp(e)),
                StudioToApp::TextInput(e) => self.call_event_handler(&Event::TextInput(e)),
                StudioToApp::TextCopy => {
                    let response = Rc::new(RefCell::new(None));
                    self.call_event_handler(&Event::TextCopy(TextClipboardEvent {
                        response: response.clone(),
                    }));
                    let text = response.borrow().clone();
                    if let Some(text) = text {
                        write_stdout_msg(&AppToStudio::SetClipboard(text));
                    }
                }
                StudioToApp::TextCut => {
                    let response = Rc::new(RefCell::new(None));
                    self.call_event_handler(&Event::TextCut(TextClipboardEvent {
                        response: response.clone(),
                    }));
                    let text = response.borrow().clone();
                    if let Some(text) = text {
                        write_stdout_msg(&AppToStudio::SetClipboard(text));
                    }
                }
                StudioToApp::MouseDown(e) => {
                    self.fingers.process_tap_count(dvec2(e.x, e.y), e.time);
                    let (window_id, pos) = self.windows.window_id_contains(dvec2(e.x, e.y));
                    let mouse_down_event = crate::event::MouseDownEvent {
                        abs: dvec2(e.x - pos.x, e.y - pos.y),
                        button: crate::event::MouseButton::from_bits_retain(e.button_raw_bits),
                        window_id,
                        modifiers: e.modifiers.into_key_modifiers(),
                        handled: std::cell::Cell::new(crate::Area::Empty),
                        time: e.time,
                    };
                    self.fingers.mouse_down(mouse_down_event.button, window_id);
                    self.call_event_handler(&Event::MouseDown(mouse_down_event));
                }
                StudioToApp::MouseMove(e) => {
                    let (window_id, pos) =
                        if let Some((_, window_id)) = self.fingers.first_mouse_button {
                            (window_id, self.windows[window_id].window_geom.position)
                        } else {
                            self.windows.window_id_contains(dvec2(e.x, e.y))
                        };
                    self.call_event_handler(&Event::MouseMove(crate::event::MouseMoveEvent {
                        abs: dvec2(e.x - pos.x, e.y - pos.y),
                        window_id,
                        modifiers: e.modifiers.into_key_modifiers(),
                        time: e.time,
                        handled: std::cell::Cell::new(crate::Area::Empty),
                    }));
                    self.fingers.cycle_hover_area(live_id!(mouse).into());
                    self.fingers.switch_captures();
                }
                StudioToApp::TweakRay(e) => {
                    let (window_id, pos) = self.windows.window_id_contains(dvec2(e.x, e.y));
                    let dpi_factor = self.windows[window_id].window_geom.dpi_factor.max(1.0);
                    let tweak_ray = crate::event::TweakRayEvent {
                        abs: dvec2(e.x - pos.x, e.y - pos.y),
                        window_id,
                        modifiers: e.modifiers.into_key_modifiers(),
                        time: e.time,
                        dpi_factor,
                        hit_widget_uids: std::cell::RefCell::new(Vec::new()),
                        hit_rect: std::cell::Cell::new(None),
                    };
                    self.call_event_handler(&Event::TweakRay(tweak_ray));
                }
                StudioToApp::MouseUp(e) => {
                    let (window_id, pos) =
                        if let Some((_, window_id)) = self.fingers.first_mouse_button {
                            (window_id, self.windows[window_id].window_geom.position)
                        } else {
                            self.windows.window_id_contains(dvec2(e.x, e.y))
                        };
                    let mouse_up_event = crate::event::MouseUpEvent {
                        abs: dvec2(e.x - pos.x, e.y - pos.y),
                        button: crate::event::MouseButton::from_bits_retain(e.button_raw_bits),
                        window_id,
                        modifiers: e.modifiers.into_key_modifiers(),
                        time: e.time,
                    };
                    let button = mouse_up_event.button;
                    self.call_event_handler(&Event::MouseUp(mouse_up_event));
                    self.fingers.mouse_up(button);
                    self.fingers.cycle_hover_area(live_id!(mouse).into());
                }
                StudioToApp::Scroll(e) => {
                    let (window_id, pos) = self.windows.window_id_contains(dvec2(e.x, e.y));
                    self.call_event_handler(&Event::Scroll(crate::event::ScrollEvent {
                        abs: dvec2(e.x - pos.x, e.y - pos.y),
                        scroll: dvec2(e.sx, e.sy),
                        window_id,
                        modifiers: e.modifiers.into_key_modifiers(),
                        handled_x: std::cell::Cell::new(false),
                        handled_y: std::cell::Cell::new(false),
                        is_mouse: e.is_mouse,
                        time: e.time,
                        phase: crate::event::ScrollPhase::None,
                    }));
                }
                StudioToApp::WindowGeomChange {
                    dpi_factor,
                    left,
                    top,
                    width,
                    height,
                    window_id,
                } => {
                    while windows.len() <= window_id {
                        windows.push(Default::default());
                    }
                    windows[window_id].created = true;
                    windows[window_id].dpi_factor = dpi_factor;
                    windows[window_id].width = width.max(1.0) as u32;
                    windows[window_id].height = height.max(1.0) as u32;
                    windows[window_id].ensure_size_defaults();

                    let window_id = CxWindowPool::from_usize(window_id);
                    if self.windows.is_valid(window_id) {
                        let old_geom = self.windows[window_id].window_geom.clone();
                        let new_geom = WindowGeom {
                            position: dvec2(left, top),
                            dpi_factor,
                            inner_size: dvec2(width, height),
                            ..Default::default()
                        };
                        self.windows[window_id].window_geom = new_geom.clone();
                        let re = WindowGeomChangeEvent {
                            window_id,
                            new_geom,
                            old_geom,
                        };
                        self.call_event_handler(&Event::WindowGeomChange(re));
                    }
                    self.redraw_all();
                }
                StudioToApp::Swapchain(shared_swapchain) => {
                    let window_id = shared_swapchain.window_id;
                    while windows.len() <= window_id {
                        windows.push(Default::default());
                    }
                    let state = &mut windows[window_id];
                    state.created = true;
                    state.width = shared_swapchain.alloc_width.max(1);
                    state.height = shared_swapchain.alloc_height.max(1);
                    state.presentable_id =
                        shared_swapchain.presentable_images.first().map(|pi| pi.id);
                    state.ensure_size_defaults();
                    self.redraw_all();
                }
                StudioToApp::RunViewFrameRequest(_) => {}
                StudioToApp::Tick => {
                    crate::studio_tick_watchdog::note_studio_tick();
                    if SignalToUI::check_and_clear_ui_signal() {
                        self.handle_termination_signal();
                        self.handle_script_signals();
                        self.call_event_handler(&Event::Signal);
                    }
                    if SignalToUI::check_and_clear_action_signal() {
                        self.handle_action_receiver();
                    }
                    self.dispatch_network_runtime_events();

                    let timer_events = self.os.stdin_timers.get_dispatch();
                    for event in timer_events {
                        self.handle_script_timer(&event);
                        self.call_event_handler(&Event::Timer(event));
                    }

                    running = self.headless_handle_platform_ops(&mut windows, true);
                    if !running {
                        break;
                    }

                    let time_now = self.os.stdin_timers.time_now();
                    if !self.new_next_frames.is_empty() {
                        self.call_next_frame_event(time_now);
                    }

                    if self.os.no_draw || self.need_redrawing() {
                        let rendered =
                            self.headless_process_draw_cycle(&mut windows, true, time_now);

                        if rendered
                            || !self.os.stdin_timers.timers.is_empty()
                            || !self.new_next_frames.is_empty()
                        {
                            write_stdout_msg(&AppToStudio::RequestAnimationFrame);
                        }
                    } else if !self.os.stdin_timers.timers.is_empty()
                        || !self.new_next_frames.is_empty()
                    {
                        write_stdout_msg(&AppToStudio::RequestAnimationFrame);
                    }
                }
                other => {
                    if self.dispatch_studio_msg(other, CxWindowPool::id_zero(), dvec2(0.0, 0.0)) {
                        break;
                    }

                    running = self.headless_handle_platform_ops(&mut windows, true);
                    if !running {
                        break;
                    }

                    let time_now = self.os.stdin_timers.time_now();
                    if !self.new_next_frames.is_empty() {
                        self.call_next_frame_event(time_now);
                    }

                    if self.os.no_draw || self.need_redrawing() {
                        let rendered =
                            self.headless_process_draw_cycle(&mut windows, true, time_now);

                        if rendered
                            || !self.os.stdin_timers.timers.is_empty()
                            || !self.new_next_frames.is_empty()
                        {
                            write_stdout_msg(&AppToStudio::RequestAnimationFrame);
                        }
                    } else if !self.os.stdin_timers.timers.is_empty()
                        || !self.new_next_frames.is_empty()
                    {
                        write_stdout_msg(&AppToStudio::RequestAnimationFrame);
                    }
                }
            }
            crate::studio_tick_watchdog::note_studio_draw_pending(self.need_redrawing());
        }
    }

    fn headless_emit_frames(
        &mut self,
        windows: &mut [HeadlessWindowState],
        send_protocol: bool,
        time_now: f64,
    ) -> bool {
        let output_dir = self.headless_output_dir();
        let mut rendered_any = false;

        // Render all passes using the real draw tree + JIT shaders
        let framebuffers = self.headless_render_all_passes(time_now);

        for (window_id, fb) in framebuffers {
            // Skip if we don't have a window state for this window
            if window_id >= windows.len() {
                continue;
            }
            let state = &mut windows[window_id];
            if !state.created {
                state.created = true;
                state.ensure_size_defaults();
            }

            let width = fb.width as u32;
            let height = fb.height as u32;

            let request_ids = if send_protocol {
                self.take_studio_screenshot_request_ids(0)
            } else {
                Vec::new()
            };
            if send_protocol && request_ids.is_empty() {
                continue;
            }

            // Instrumentación H0-bis: el volcado a PNG es coste de la SONDA, no
            // del escritorio (en ATLAS el present es un blit RGBA a la surface).
            // Se cronometra aparte y se puede desactivar con
            // `MAKEPAD_HEADLESS_NO_PNG=1` para medir sin ese ruido.
            let profile_on = std::env::var("MAKEPAD_HEADLESS_PROFILE").is_ok();
            let conv_start = std::time::Instant::now();
            // Buffer de salida PERSISTENTE: lo de fuera del daño no se vuelve a
            // convertir, que es justo el punto. Antes se reservaba un `Vec` nuevo
            // (3,7 MB a 1280×720) y se convertía la pantalla entera en cada frame.
            let rgba = {
                use std::cell::RefCell;
                thread_local! {
                    static SALIDA: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
                }
                SALIDA.with(|b| {
                    let mut b = b.borrow_mut();
                    fb.to_rgba8_into(&mut b, crate::os::headless::virtual_gpu::headless_clip_rect());
                    b.clone()
                })
            };
            let conv_ms = conv_start.elapsed().as_secs_f64() * 1000.0;
            if std::env::var("MAKEPAD_HEADLESS_NO_PNG").is_ok() {
                if profile_on {
                    crate::log!(
                        "[headless][profile] to_rgba8(present)={:.1}ms png=omitido bytes={}",
                        conv_ms,
                        rgba.len()
                    );
                }
                state.frame_id += 1;
                rendered_any = true;
                continue;
            }
            let png_start = std::time::Instant::now();
            let png = match encode_png_rgba(width, height, &rgba) {
                Ok(png) => png,
                Err(err) => {
                    crate::error!(
                        "headless png encode failed for window {} frame {}: {}",
                        window_id,
                        state.frame_id,
                        err
                    );
                    continue;
                }
            };
            if profile_on {
                crate::log!(
                    "[headless][profile] to_rgba8(present)={:.1}ms png_encode={:.1}ms",
                    conv_ms,
                    png_start.elapsed().as_secs_f64() * 1000.0
                );
            }

            let png_path = output_dir.join(format!(
                "window_{window_id}_frame_{:06}.png",
                state.frame_id
            ));
            if let Err(err) = std::fs::write(&png_path, &png) {
                crate::error!(
                    "headless frame write failed for `{}`: {}",
                    png_path.display(),
                    err
                );
                continue;
            }

            if send_protocol {
                write_stdout_msg(&AppToStudio::Screenshot(ScreenshotResponse {
                    request_ids,
                    png,
                    width,
                    height,
                }));
                let target_id = if let Some(id) = state.presentable_id {
                    id
                } else {
                    let id = PresentableImageId::alloc();
                    state.presentable_id = Some(id);
                    id
                };
                write_stdout_msg(&AppToStudio::DrawCompleteAndFlip(PresentableDraw {
                    window_id,
                    target_id,
                    width,
                    height,
                }));
            } else {
                crate::log!(
                    "headless frame written: {} ({}x{})",
                    png_path.display(),
                    width,
                    height
                );
            }

            state.frame_id += 1;
            rendered_any = true;
        }

        rendered_any
    }

    fn headless_output_dir(&mut self) -> PathBuf {
        if let Some(path) = &self.os.frame_dir {
            return path.clone();
        }
        let path = std::env::var("MAKEPAD_HEADLESS_OUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        if let Err(err) = std::fs::create_dir_all(&path) {
            crate::error!(
                "failed to create headless frame output dir `{}`: {}",
                path.display(),
                err
            );
        }
        self.os.frame_dir = Some(path.clone());
        path
    }

    fn headless_handle_platform_ops(
        &mut self,
        windows: &mut Vec<HeadlessWindowState>,
        send_protocol: bool,
    ) -> bool {
        while let Some(op) = self.platform_ops.pop_front() {
            match op {
                CxOsOp::CreateWindow(window_id) => {
                    while window_id.id() >= windows.len() {
                        windows.push(Default::default());
                    }

                    let window = &mut self.windows[window_id];
                    // Tamaño de la ventana headless.
                    //
                    // `MAKEPAD_HEADLESS_SIZE=ANCHOxALTO` (ATLAS/H0) fuerza el
                    // tamaño lógico de TODA ventana creada, ignorando el que
                    // pida la app. Sirve para medir siempre a la misma
                    // resolución (p. ej. 1280x720) sin tocar el código de la
                    // app ni el de Brasa, que es de sólo lectura para este
                    // experimento. Sin la variable, se respeta lo que pida la
                    // app y, si no pide nada, el 1920x1080 de siempre.
                    let inner_size = headless_forced_window_size().unwrap_or_else(|| {
                        window
                            .create_inner_size
                            .unwrap_or_else(|| dvec2(1920.0, 1080.0))
                    });
                    let position = window.create_position.unwrap_or_else(|| dvec2(0.0, 0.0));
                    let dpi_factor = configured_headless_dpi();

                    let state = &mut windows[window_id.id()];
                    state.created = true;
                    state.dpi_factor = dpi_factor;
                    state.width = inner_size.x.max(1.0) as u32;
                    state.height = inner_size.y.max(1.0) as u32;

                    window.is_created = true;
                    window.window_geom.position = position;
                    window.window_geom.inner_size = inner_size;
                    window.window_geom.outer_size = inner_size;
                    window.window_geom.dpi_factor = dpi_factor;
                    if send_protocol {
                        write_stdout_msg(&AppToStudio::CreateWindow {
                            window_id: window_id.id(),
                            kind_id: window.kind_id,
                        });
                    }
                    self.redraw_all();
                }
                CxOsOp::CreatePopupWindow {
                    window_id,
                    parent_window_id,
                    position,
                    size,
                    grab_keyboard,
                } => {
                    while window_id.id() >= windows.len() {
                        windows.push(Default::default());
                    }
                    let state = &mut windows[window_id.id()];
                    state.created = true;
                    state.width = size.x.max(1.0) as u32;
                    state.height = size.y.max(1.0) as u32;
                    state.ensure_size_defaults();

                    let window = &mut self.windows[window_id];
                    window.is_created = true;
                    window.window_geom.position = position;
                    window.window_geom.inner_size = size;
                    window.window_geom.outer_size = size;
                    window.is_popup = true;
                    window.popup_parent = Some(parent_window_id);
                    window.popup_position = Some(position);
                    window.popup_size = Some(size);
                    window.popup_grab_keyboard = grab_keyboard;
                    self.redraw_all();
                }
                CxOsOp::ResizeWindow(window_id, size) => {
                    if self.windows.is_valid(window_id) {
                        self.windows[window_id].window_geom.inner_size = size;
                    }
                    while window_id.id() >= windows.len() {
                        windows.push(Default::default());
                    }
                    windows[window_id.id()].created = true;
                    windows[window_id.id()].width = size.x.max(1.0) as u32;
                    windows[window_id.id()].height = size.y.max(1.0) as u32;
                    windows[window_id.id()].ensure_size_defaults();
                    self.redraw_all();
                }
                CxOsOp::SetCursor(cursor) => {
                    if send_protocol {
                        write_stdout_msg(&AppToStudio::SetCursor(cursor.into()));
                    }
                }
                CxOsOp::StartTimer {
                    timer_id,
                    interval,
                    repeats,
                } => {
                    self.os
                        .stdin_timers
                        .timers
                        .insert(timer_id, PollTimer::new(interval, repeats));
                }
                CxOsOp::StopTimer(timer_id) => {
                    self.os.stdin_timers.timers.remove(&timer_id);
                }
                CxOsOp::CopyToClipboard(content) => {
                    if send_protocol {
                        write_stdout_msg(&AppToStudio::SetClipboard(content));
                    }
                }
                CxOsOp::Quit => {
                    return false;
                }
                // Track selection is currently implemented on Linux GStreamer only.
                CxOsOp::SelectVideoTrack(_, _) | CxOsOp::SelectAudioTrack(_, _) => {}
                _ => {}
            }
        }
        true
    }
}

impl CxOsApi for Cx {
    fn init_cx_os(&mut self) {
        self.os.start_time = Some(Instant::now());
        self.os.no_draw = crate::app_main::should_disable_headless_draw_from_args();
        self.os.draw_cycles = crate::app_main::headless_draw_cycles_from_args();
        if let Some(item) = std::option_env!("MAKEPAD_PACKAGE_DIR") {
            self.package_root = Some(item.to_string());
        }
        self.native_load_dependencies();
    }

    fn spawn_thread<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        std::thread::spawn(f);
    }

    fn seconds_since_app_start(&self) -> f64 {
        Instant::now()
            .duration_since(self.os.start_time.unwrap_or_else(Instant::now))
            .as_secs_f64()
    }

    fn open_url(&mut self, _url: &str, _in_place: OpenUrlInPlace) {
        crate::warning!("open_url is ignored in headless mode");
    }
}

fn write_stdout_msg(msg: &AppToStudio) {
    let _ = io::stdout().write_all(msg.to_json().as_bytes());
    let _ = io::stdout().write_all(b"\n");
    let _ = io::stdout().flush();
}
