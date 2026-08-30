//! # sonda_teclado_movil — ¿sube el teclado de un móvil en el navegador?
//!
//! Un campo de búsqueda y nada más. Existe porque el fallo que mide **no se
//! reproduce en el ordenador donde se programa**: en un navegador de escritorio
//! el teclado «funciona» siempre, porque hay uno físico y porque el foco lo pide
//! `mousedown`. En un móvil no llega ningún `mousedown` —los manejadores
//! táctiles hacen `preventDefault()`— y el teclado no sube nunca.
//!
//! ## Cómo se usa
//!
//! ```text
//! cargo makepad wasm --no-threads --port=8099 run -p makepad-sonda-teclado-movil
//! ```
//!
//! y se abre en el teléfono. Con `adb reverse tcp:8099 tcp:8099`, la dirección
//! desde el móvil es `http://localhost:8099`.
//!
//! ## Qué mirar, y por qué NO basta con mirar la pantalla
//!
//! Que salga el teclado se ve, pero «se ve» no es una medida. Android lo dice:
//!
//! ```text
//! adb shell dumpsys input_method | grep -E "mInputShown|mServedView"
//! ```
//!
//! `mInputShown=true` es el teclado desplegado, y lo dice el sistema, no
//! nosotros. Es el único oráculo que no comparte premisa con el código que se
//! está probando.
//!
//! 🔴 **Y antes de creerse un `false`, hay que hacer el CONTROL**: el mismo
//! `adb shell input tap` sobre un `<input>` HTML nativo tiene que levantar el
//! teclado. Sin eso, un `false` no distingue «el arreglo no funciona» de «así de
//! tocar no vale» — y las dos cosas se ven igual.
//!
//! ⚠ Dos trampas más, las dos medidas el 2026-08-30:
//!
//! - **Chrome congela las pestañas de fondo.** Al abrir una segunda pestaña, la
//!   primera deja de contestar al protocolo DevTools, y eso se lee igual que
//!   «la página se ha colgado». Se mide sobre la que está delante.
//! - **El primer toque después de cargar puede perderse**, mientras la escena
//!   se asienta. Si el primero no hace nada, repetir antes de concluir.
//!
//! ## Verificado
//!
//! - **S9+** (SM-G965F): teclado arriba, tecla de BUSCAR, texto recibido, se
//!   baja con atrás y no rebota, y vuelve a subir al tocar. Tres ciclos.
//! - **Poco F7 Pro** (24117RK2CG, Chrome 151): igual, y con el autocompletado
//!   del sistema encima.
//!
//! Y en el navegador, la mitad que explica el porqué:
//!
//! ```js
//! document.activeElement.className   // «cx_webgl_textinput» = tiene el foco
//! document.activeElement.getAttribute('inputmode')  // «search» = teclado de buscar
//! ```

use makepad_widgets::*;

script_mod! {
    use mod.prelude.widgets.*
    startup() do #(SondaTecladoMovil::script_component(vm)){
        ui: Root { main_window := Window {
            window.title: "sonda teclado movil"
            body +: { fondo := SolidView {
                width: Fill height: Fill flow: Down
                padding: 20 spacing: 14
                show_bg: true draw_bg +: { color: #222 }

                titulo := Label {
                    height: Fit
                    text: "Toca el campo. En un movil debe subir el teclado."
                    draw_text +: { color: #eee }
                }

                // Un buscador, que es el caso que lo destapo: ademas de subir,
                // el teclado deberia traer tecla de BUSCAR y no de salto de
                // linea, que es lo que dan `input_mode` y `return_key_type`.
                //
                // ⚠ Van PLANOS, no anidados bajo `soft_keyboard`: el DSL ignora
                // una propiedad que no existe con un error en la consola que es
                // facil no ver, y entonces el campo sale con el teclado
                // generico y parece que el arreglo no funciona.
                buscar := TextInput {
                    width: Fill height: 44
                    empty_text: "buscar personaje"
                    input_mode: Search
                    return_key_type: Search
                    submit_on_enter: true
                }

                eco := Label {
                    height: Fit
                    text: "(sin escribir nada todavia)"
                    draw_text +: { color: #8f8 }
                }
            } }
        } }
    }
}

/// La sonda: un campo y un eco de lo que se teclea.
#[derive(Script, ScriptHook)]
pub struct SondaTecladoMovil {
    #[live]
    ui: WidgetRef,
}

impl MatchEvent for SondaTecladoMovil {
    fn handle_actions(&mut self, cx: &mut Cx, actions: &Actions) {
        // El eco es la prueba de que las teclas LLEGAN, que es distinto de que
        // el teclado se vea: en móvil se puede tener lo segundo sin lo primero.
        if let Some(t) = self.ui.text_input(cx, ids!(buscar)).changed(actions) {
            self.ui
                .label(cx, ids!(eco))
                .set_text(cx, &format!("escrito: «{t}»"));
        }
    }
}

impl AppMain for SondaTecladoMovil {
    // Sin esto compila y revienta AL ARRANCAR, no al construir: el DSL de
    // makepad es un interprete, asi que `cargo build` en verde no dice nada
    // sobre si la aplicacion levanta.
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        makepad_widgets::script_mod(vm);
        self::script_mod(vm)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        self.match_event(cx, event);
        self.ui.handle_event(cx, event, &mut Scope::empty());
    }
}

app_main!(SondaTecladoMovil);
