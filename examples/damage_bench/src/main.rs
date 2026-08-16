//! Banco del repintado parcial: una escena que se ensucia SOLO EN UN TROZO.
//!
//! # Por qué existe
//!
//! El cálculo de daño (`platform/src/os/headless/damage.rs`) no se puede verificar
//! con una aplicación quieta: si nada cambia, no hay frames; y si se fuerza el
//! repintado con `MAKEPAD_HEADLESS_FORCE_REDRAW`, el daño es siempre la pantalla
//! entera y no se ejerce ni una línea del cálculo.
//!
//! Y el fallo que hay que cazar **sólo aparece en movimiento**: si el daño se deja
//! algo fuera, lo que queda en pantalla son restos del frame anterior — un
//! fantasma. Una captura estática sale perfecta, que es justo lo que suele
//! mirarse.
//!
//! Esta escena tiene las dos mitades que hacen falta:
//!
//! - Un **fondo grande y quieto** (una rejilla de 240 etiquetas). Si el daño
//!   funciona, no se vuelve a tocar ni un píxel de aquí después del primer frame.
//! - Un **contador pequeño que cambia solo**, con `NextFrame`. Es lo único que
//!   debería ensuciarse.
//! - Y un **bloque que se ENCOGE** cada pocos frames. Esta es la que caza el fallo
//!   de verdad: al menguar, su rectángulo nuevo NO cubre donde estaba, así que si
//!   el daño no incluye la posición ANTERIOR, queda el resto pintado. Es el fallo
//!   clásico del repintado parcial y el que un contador que sólo cambia de cifra
//!   nunca destapa.
//!
//! # Cómo se usa como oráculo
//!
//! Dos corridas y una comparación:
//!
//! ```text
//! MAKEPAD=headless MAKEPAD_HEADLESS_DAMAGE=1 ... --draws=30   # con daño
//! MAKEPAD=headless                            ... --draws=30   # entero siempre
//! ```
//!
//! Si el daño es correcto, las dos series de PNG son **idénticas píxel a píxel**.
//! Cualquier diferencia es, literalmente, lo que el daño se dejó fuera. El script
//! `tools/verificar-dano.sh` hace las dos corridas y la comparación.

pub use makepad_widgets;

use makepad_widgets::*;

app_main!(App);

script_mod! {
    use mod.prelude.widgets.*
    let state = {
        tick: 0,
        // Ancho del bloque que mengua. Se recalcula por frame en Rust; aquí sólo
        // se guarda para que el DSL lo lea.
        ancho: 600.0,
    }
    mod.state = state
    startup() do #(App::script_component(vm)){
        ui: Root{
            on_startup:||{
                ui.fondo.render()
                ui.contador.render()
                ui.menguante.render()
            }
            main_window := Window{
                window.inner_size: vec2(1000, 700)
                window.title: "damage bench"
                body +: {
                    raiz := SolidView{
                        width: Fill height: Fill
                        flow: Down
                        show_bg: true
                        draw_bg +: { color: #1E1D1B }

                        // ── Lo que NO debe repintarse nunca más ──────────────
                        // Una rejilla densa y quieta: si el daño se calcula mal
                        // por exceso, esto se rasteriza otra vez y el ahorro
                        // medido se desploma. Es a la vez decorado y medidor.
                        fondo := SolidView{
                            width: Fill height: 420.0
                            flow: Right
                            flow_wrap: true
                            padding: 6.0
                            spacing: 4.0
                            show_bg: true
                            draw_bg +: { color: #141312 }
                            on_render: ||{
                                for i in 0..240 {
                                    Label{
                                        text: "quieto " + i
                                        draw_text +: { color: #A8A49C, text_style.font_size: 9.0 }
                                    }
                                }
                            }
                        }

                        // ── Lo único que cambia por frame ───────────────────
                        contador := SolidView{
                            // `new_batch` = draw list PROPIA. Sin esto, este View
                            // dibuja dentro de la lista del padre y el daño sale
                            // con la granularidad del padre: la ventana entera.
                            new_batch: true
                            width: Fill height: 60.0
                            padding: 10.0
                            show_bg: true
                            draw_bg +: { color: #28272A }
                            on_render: ||{
                                Label{
                                    text: "tick " + state.tick
                                    draw_text +: { color: #FAC775, text_style.font_size: 22.0 }
                                }
                            }
                        }

                        // ── El que MENGUA: caza los fantasmas ───────────────
                        menguante := SolidView{
                            new_batch: true
                            width: Fill height: Fit
                            padding: 8.0
                            show_bg: true
                            draw_bg +: { color: #1E1D1B }
                            on_render: ||{
                                SolidView{
                                    width: state.ancho
                                    height: 90.0
                                    show_bg: true
                                    draw_bg +: { color: #D85A30 }
                                }
                            }
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
    /// Frame actual. En Rust y no sólo en el DSL porque de aquí sale la anchura,
    /// y el cálculo debe ser el mismo en cada corrida: el oráculo compara dos
    /// ejecuciones píxel a píxel, así que **nada puede depender del reloj**.
    #[rust]
    tick: u64,
}

impl MatchEvent for App {
    fn handle_startup(&mut self, cx: &mut Cx) {
        cx.new_next_frame();
    }

    /// Un paso por frame. Es determinista a propósito: `tick` avanza de uno en
    /// uno y la anchura sale de una tabla, no del tiempo transcurrido. Si esto
    /// dependiera del reloj, las dos corridas del oráculo divergirían por sí
    /// solas y toda diferencia sería ruido.
    fn handle_next_frame(&mut self, cx: &mut Cx, _e: &NextFrameEvent) {
        self.tick += 1;
        // Escalón de anchura cada 5 frames: 600 → 480 → 360 → 240 → 120 → 600…
        // Menguar es el caso que importa; el salto de vuelta a 600 ejerce el
        // contrario (crecer), que no deja fantasmas pero sí debe repintarse.
        let paso = (self.tick / 5) % 5;
        let ancho = 600.0 - 120.0 * paso as f64;
        let tick = self.tick as f64;
        script_eval!(cx, {
            mod.state.tick = #(tick)
            mod.state.ancho = #(ancho)
            // Se ensucian SÓLO estos dos. El fondo no se toca: es el testigo.
            ui.contador.render()
            ui.menguante.render()
        });
        cx.new_next_frame();
    }
}

impl AppMain for App {
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        crate::makepad_widgets::script_mod(vm);
        self::script_mod(vm)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        self.match_event(cx, event);
        self.ui.handle_event(cx, event, &mut Scope::empty());
    }
}
