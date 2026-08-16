// build.rs de `makepad-draw`
//
// POR QUÉ EXISTE (añadido para el hito H0 de ATLAS):
//
// El backend `headless` (rasterizador por software, sin GPU) se activa con la
// variable de entorno `MAKEPAD=headless`, que `platform/build.rs` traduce a
// `--cfg headless`. Pero ese cfg es POR CRATE: sólo existía dentro de
// `makepad-platform`. `makepad-draw` también necesita saberlo, porque
// `draw/src/shader/draw_text.rs` tiene DOS implementaciones completas del
// shader de texto:
//
//   - la ruta "slug" (`#[cfg(any(target_os = "linux", target_os = "windows"))]`),
//     que depende de que el shader esté listo en el contexto GL/D3D real
//     (`Cx::is_draw_shader_window_ready`, que sólo existe en los backends
//     OpenGL y D3D11), y
//   - la ruta genérica (macOS y demás), que no depende de ningún contexto de GPU.
//
// Compilando en Linux con `headless` no hay backend OpenGL, así que la ruta
// "slug" no enlaza. Con este cfg propagado, linux+headless toma la ruta
// genérica —la misma con la que se desarrolló el backend headless en macOS—.
//
// Sin este fichero, `cfg(headless)` en `makepad-draw` se evaluaría siempre como
// falso (y además emitiría el warning `unexpected_cfgs`).
use std::env;

fn main() {
    // Declaramos el cfg para que rustc no lo marque como desconocido.
    println!("cargo:rustc-check-cfg=cfg(headless)");
    // Recompilar si cambia MAKEPAD (p. ej. al alternar headless / nativo).
    println!("cargo:rerun-if-env-changed=MAKEPAD");
    if let Ok(configs) = env::var("MAKEPAD") {
        for config in configs.split(['+', ',']) {
            if config == "headless" {
                println!("cargo:rustc-cfg=headless");
            }
        }
    }
}
