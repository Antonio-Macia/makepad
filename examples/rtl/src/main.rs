//! # rtl — the right-to-left layout mirror, and the cases that break it
//!
//! ```bash
//! cargo run -p makepad-example-rtl        # R = right-to-left, L = back
//! ```
//!
//! ## Why this example exists
//!
//! `Cx::reading_direction` mirrors a row's children so an Arabic, Hebrew,
//! Persian or Urdu interface reads the right way round. It was built without an
//! example that turned it on, and it cost twice:
//!
//! 1. The first version dropped **every `width: Fill` child** — a fill has no
//!    width while the row is being walked, so the mirror reflected zeros. Fixed
//!    by mirroring in `end_turtle_with_guard` instead, which is the first moment
//!    a fill has a width.
//! 2. That fix was real and **incomplete**: a `Button` at `width: Fill` still
//!    came out with no text at all. Ten green turtle tests said otherwise,
//!    because none of them opens a window.
//!
//! So this file is not "a mirrored row". It is the shapes that actually broke,
//! each next to the one that works, which is what makes a diagnosis take minutes
//! instead of a day.
//!
//! ## What to look at
//!
//! - **A · Button at Fill** — the broken one. Both labels must stay visible and
//!   centred in their boxes after pressing `R`.
//! - **B · View at Fill with a Label inside** — the control. It has always
//!   worked, and it is what proves the row's own reflection is fine.
//! - **C · Fit** — the case that never broke. If this one goes, the good part
//!   went with it.
//! - **D · `mirror: false`** — must NOT flip: `▶` points where time goes, not
//!   where the text goes.
//! - And pressing `L` must leave the screen exactly as it started. A mirror that
//!   is not an involution leaves a trail, and the trail accumulates.

pub use makepad_widgets;

use makepad_widgets::*;

app_main!(App);

script_mod! {
    use mod.prelude.widgets.*

    startup() do #(App::script_component(vm)){
        ui: Root{
            main_window := Window{
                // Without a title the window is called "Makepad" and cannot be
                // found by name, so no automated capture can verify it.
                window.title: "Makepad · RTL"
                window.inner_size: vec2(620, 400)
                body +: {
                    root := SolidView{
                        width: Fill height: Fill flow: Down spacing: 10 padding: 12
                        show_bg: true
                        draw_bg +: { color: #2b2b2b }

                        help := Label{ text: "R = right-to-left · L = left-to-right" }
                        echo := Label{ width: Fill }

                        // A — the broken one: Button at Fill.
                        row_a := View{
                            width: Fill height: Fit flow: Right spacing: 8
                            a1 := Button{ width: Fill, text: "A left button" }
                            a2 := Button{ width: Fill, text: "A right button" }
                        }

                        // B — the control: View at Fill with a Label inside.
                        row_b := View{
                            width: Fill height: Fit flow: Right spacing: 8
                            b1 := SolidView{
                                width: Fill height: Fit padding: 6
                                show_bg: true
                                draw_bg +: { color: #3a3a3a }
                                b1l := Label{ text: "B left view" }
                            }
                            b2 := SolidView{
                                width: Fill height: Fit padding: 6
                                show_bg: true
                                draw_bg +: { color: #3a3a3a }
                                b2l := Label{ text: "B right view" }
                            }
                        }

                        // E — one single Fill next to a Fit. Halves the search
                        //     space: if the label goes missing here too, the fault
                        //     is not two children swapping ranges.
                        row_e := View{
                            width: Fill height: Fit flow: Right spacing: 8
                            e0 := Label{ text: "E:" }
                            e1 := Button{ width: Fill, text: "E only fill" }
                        }

                        // F — THE OPPOSITE CASE, and it must keep working: a Label
                        //     at width: Fill. Its `align.x = 0` DOES mean "at the
                        //     start of the reading", so in RTL the Arabic must sit
                        //     against the RIGHT edge. This is what the alignment
                        //     flip was built for; if F breaks, the fix for A broke it.
                        row_f := View{
                            width: Fill height: Fit flow: Right spacing: 8
                            f0 := Label{ width: Fill, text: "F عربية نص" }
                        }

                        // C — Fit: never broke, and must keep not breaking.
                        row_c := View{
                            width: Fill height: Fit flow: Right spacing: 8
                            c0 := Label{ text: "C:" }
                            c1 := Button{ text: "one" }
                            c2 := Button{ text: "two" }
                            c3 := Button{ text: "three" }
                        }

                        // D — the escape hatch: this must not flip.
                        row_d := View{
                            width: Fill height: Fit flow: Right spacing: 8
                            mirror: false
                            d0 := Label{ text: "D mirror:false —" }
                            d1 := Button{ text: "|<" }
                            d2 := Button{ text: ">" }
                            d3 := Button{ text: ">|" }
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
        if let Event::KeyDown(ke) = event {
            if matches!(ke.key_code, KeyCode::KeyR | KeyCode::KeyL) {
                let rtl = ke.key_code == KeyCode::KeyR;
                cx.reading_direction = if rtl {
                    ReadingDirection::Rtl
                } else {
                    ReadingDirection::Ltr
                };
                // The direction is captured when each turtle is created, so a
                // change is not visible until the tree is drawn again.
                self.ui.redraw(cx);
                let t = if rtl {
                    "RTL — rows A, B and C flipped and STILL VISIBLE; D unchanged"
                } else {
                    "LTR — must look exactly like it did on startup"
                };
                self.ui.label(cx, ids!(echo)).set_text(cx, t);
            }
        }
        self.match_event(cx, event);
        self.ui.handle_event(cx, event, &mut Scope::empty());
    }
}
