//! Tabla de shaders precompilados AOT del backend `headless`.
//!
//! # Qué problema resuelve
//!
//! El backend `headless` (rasterizador por software) no tiene GPU, así que
//! traduce cada shader de Makepad a código Rust y lo compila **en tiempo de
//! ejecución** llamando a `rustc` y cargando el `.so` con `dlopen`
//! (ver [`super::jit`]). Ese camino exige dos cosas que no siempre existen:
//! un compilador de Rust instalado y un cargador dinámico.
//!
//! Dentro de **ATLAS OS** —el sistema operativo bare-metal para el que se está
//! construyendo este backend— no hay ninguna de las dos, y no las habrá: el
//! JIT ahí es sencillamente imposible. Además, aun teniéndolas, compilar los
//! shaders al arrancar es caro: medido sobre una pantalla real de Brasa, los
//! 50 shaders de la escena cuestan ~22 s en el primer frame.
//!
//! La salida es precompilarlos **en el build del host**: se ejecuta la app una
//! vez con el JIT, que deja un `shader_<hash>/lib.rs` por shader, y la
//! siguiente compilación los empotra en el binario (`MAKEPAD_HEADLESS_AOT_DIR`,
//! ver `platform/build.rs`). Este módulo es el lado de runtime de eso.
//!
//! # Por qué la tabla resuelve *por nombre*
//!
//! Podría exponerse una struct con once campos tipados, uno por punto de
//! entrada. Se ha preferido un resolvedor `fn(&str) -> Option<*const ()>`
//! porque es **exactamente la forma de `dlsym`**: así el resto del backend
//! (`raster.rs`, `shader.rs`) sigue escribiendo `module.symbol::<F>("nombre")`
//! sin enterarse de si detrás hay una biblioteca dinámica o código empotrado.
//! El AOT es un reemplazo transparente, no un segundo camino a mantener en
//! paralelo. El precio —una comparación de cadenas por símbolo— se paga una
//! vez por draw-call, no por píxel, y es ruido frente a los ~150 ms de
//! rasterizado de un frame.
//!
//! # Seguridad
//!
//! Los punteros de la tabla apuntan a funciones del propio binario: son
//! válidos durante toda la vida del proceso (`'static`), a diferencia de los
//! de `dlopen`, que mueren con el `dlclose`. Sigue siendo responsabilidad de
//! quien llama transmutarlos a la firma correcta; esa es la misma condición
//! que ya imponía el camino `dlsym` y por eso el interfaz no empeora.

/// Un shader precompilado y empotrado en el binario.
///
/// Se construyen exclusivamente desde el fichero generado por
/// `platform/build.rs`; no hay razón para crearlos a mano.
pub struct AotShader {
    /// Hash del código fuente del shader, tal y como lo calcula
    /// `super::shader`. Es la clave de búsqueda: el mismo valor que el JIT usa
    /// para nombrar el directorio `shader_<hash>`.
    pub source_hash: u64,
    /// Resolvedor de símbolos del módulo, equivalente a `dlsym`. Devuelve
    /// `None` para los puntos de entrada que ese shader concreto no emite
    /// (p. ej. `makepad_headless_rcx_frag_offset` sólo existe si el shader lee
    /// el framebuffer).
    pub resolve: fn(&str) -> Option<*const ()>,
}

// Fichero generado por `platform/build.rs`. Define `AOT_SHADERS`. Cuando
// `MAKEPAD_HEADLESS_AOT_DIR` no está puesto —el caso por defecto— la tabla
// está vacía y todo el mecanismo es coste cero.
include!(concat!(env!("OUT_DIR"), "/headless_aot_gen.rs"));

/// Busca un shader precompilado por el hash de su fuente.
///
/// Búsqueda lineal a propósito: la tabla tiene decenas de entradas y se
/// consulta una sola vez por shader (al compilarlo), no por frame. Un mapa
/// ordenado añadiría dependencias y código generado sin ganancia medible.
pub fn lookup(source_hash: u64) -> Option<&'static AotShader> {
    AOT_SHADERS.iter().find(|s| s.source_hash == source_hash)
}

/// Número de shaders empotrados en este binario.
///
/// Existe para que los tests y las herramientas de diagnóstico puedan
/// distinguir "compilado sin AOT" (0) de "compilado con AOT" (>0) sin exponer
/// la tabla entera.
pub fn embedded_count() -> usize {
    AOT_SHADERS.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// La tabla debe ser consultable siempre, incluso vacía: un binario
    /// compilado sin `MAKEPAD_HEADLESS_AOT_DIR` tiene que seguir arrancando y
    /// caer al JIT, no romper.
    #[test]
    fn lookup_on_empty_or_populated_table_never_panics() {
        assert!(lookup(0xdead_beef_dead_beef).is_none() || embedded_count() > 0);
    }

    /// Invariante del generador: no puede haber dos entradas con el mismo
    /// hash, porque la búsqueda devolvería la primera en silencio y el shader
    /// equivocado se usaría para pintar.
    #[test]
    fn embedded_hashes_are_unique() {
        let mut seen: Vec<u64> = AOT_SHADERS.iter().map(|s| s.source_hash).collect();
        let total = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), total, "hashes de shader AOT duplicados");
    }

    /// Todo shader empotrado tiene que exponer, como mínimo, los tres puntos
    /// de entrada sin los cuales el rasterizador no puede usarlo: versión,
    /// vértice y fragmento. Si el generador dejara de emitir alguno, el
    /// backend lo descartaría en silencio (`continue` en `raster.rs`) y la
    /// pantalla saldría incompleta sin ningún error.
    #[test]
    fn embedded_shaders_expose_mandatory_entry_points() {
        for shader in AOT_SHADERS {
            for name in [
                "makepad_headless_shader_version",
                "makepad_headless_vertex",
                "makepad_headless_fragment",
            ] {
                assert!(
                    (shader.resolve)(name).is_some(),
                    "shader AOT {:016x} no expone `{name}`",
                    shader.source_hash
                );
            }
        }
    }

    /// Un nombre inventado debe fallar, no devolver un puntero cualquiera.
    #[test]
    fn unknown_symbol_resolves_to_none() {
        for shader in AOT_SHADERS {
            assert!((shader.resolve)("makepad_headless_no_existe").is_none());
        }
    }
}
