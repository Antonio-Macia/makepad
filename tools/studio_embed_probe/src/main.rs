//! `studio_embed_probe` — a deliberately tiny app whose only job is to say,
//! out loud and over the studio websocket, **whether it ever got to draw**.
//!
//! Why it exists: an app embedded in a foreign host (a DAW, a VST3/CLAP
//! plugin window, Studio's own run view) runs with `--stdin-loop` and has no
//! window loop of its own. In that mode the ONLY place that turns a dirty
//! widget tree into a frame is the `StudioToApp::Tick` branch of the stdin
//! event loop. A host that never sends `Tick` therefore produces a perfectly
//! silent failure: socket connected, `CreateWindow` acknowledged, zero errors,
//! zero pixels.
//!
//! Silence cannot be measured. So this probe emits one line per draw
//! (`PROBE-DRAW <n>`) through `log!`, which in studio mode travels to the host
//! as an `AppToStudio::LogItem`. Counting those lines at the host tells you,
//! without a GPU and without a swapchain, whether `call_draw_event` ever ran.
//!
//! Run it against `studio_embed_host` (same folder), which can be told to
//! behave like a correct host or like the broken ones we are hunting.

pub use makepad_widgets;

use makepad_widgets::*;
use std::sync::atomic::{AtomicU64, Ordering};

app_main!(App);

/// Draws seen so far. A static (rather than a struct field) so the probe's
/// `App` stays a plain `#[live]`-only component and the count survives any
/// live-reload of the component itself.
static DRAWS: AtomicU64 = AtomicU64::new(0);
/// Startup events seen. Reported so a host can tell "the app never started"
/// apart from "the app started and never drew" — two very different bugs that
/// otherwise both look like an empty window.
static STARTUPS: AtomicU64 = AtomicU64::new(0);

script_mod! {
    use mod.prelude.widgets.*

    startup() do #(App::script_component(vm)){
        ui: Root{
            main_window := Window{
                window.inner_size: vec2(420, 220)
                body +: {
                    View{
                        width: Fill
                        height: Fill
                        flow: Down
                        align: Center
                        Label{
                            text: "embed probe"
                            draw_text.text_style.font_size: 24
                        }
                    }
                }
            }
        }
    }
}

#[derive(Script, ScriptHook)]
pub struct App {
    #[live]
    ui: WidgetRef,
}

impl MatchEvent for App {}

impl AppMain for App {
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        crate::makepad_widgets::script_mod(vm);
        self::script_mod(vm)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        match event {
            Event::Startup => {
                let n = STARTUPS.fetch_add(1, Ordering::Relaxed) + 1;
                log!("PROBE-STARTUP {}", n);
            }
            Event::Draw(_) => {
                let n = DRAWS.fetch_add(1, Ordering::Relaxed) + 1;
                log!("PROBE-DRAW {}", n);
            }
            _ => {}
        }
        self.match_event(cx, event);
        self.ui.handle_event(cx, event, &mut Scope::empty());
    }
}
