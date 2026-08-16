use std::ffi::{c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct HeadlessShaderJit {
    root_dir: PathBuf,
}

pub struct HeadlessJitOutput {
    pub dylib_path: PathBuf,
    pub module: Option<HeadlessLoadedModule>,
    pub shader_version: Option<u32>,
    pub load_error: Option<String>,
}

impl Default for HeadlessShaderJit {
    fn default() -> Self {
        let root_dir = std::env::var("MAKEPAD_HEADLESS_JIT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_jit_root_dir());
        Self { root_dir }
    }
}

fn default_jit_root_dir() -> PathBuf {
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        return PathBuf::from(target_dir).join("makepad-headless-jit");
    }
    if let Ok(cwd) = std::env::current_dir() {
        return cwd.join("target").join("makepad-headless-jit");
    }
    PathBuf::from("target/makepad-headless-jit")
}

impl HeadlessShaderJit {
    pub fn compile_and_load(
        &self,
        source_hash: u64,
        source: &str,
    ) -> Result<HeadlessJitOutput, String> {
        let shader_dir = self.root_dir.join(format!("shader_{source_hash:016x}"));
        let cached_path =
            shader_dir.join(format!("shader_{source_hash:016x}.{}", dylib_extension()));

        // The dylib is content-addressed by the hash of the source that made it,
        // so one left by an earlier run is exactly what rustc would produce now.
        // Reuse it: this compiles EVERY shader with `rustc -O` at startup, which
        // costs tens of seconds per process — and a headless test suite starts a
        // fresh process for every test. Anything unloadable (truncated by a killed
        // run, built by a different toolchain) just falls through and recompiles.
        if cached_path.is_file() {
            if let Ok(loaded) = HeadlessLoadedModule::load(&cached_path) {
                if let Ok(version) = loaded.shader_version() {
                    return Ok(HeadlessJitOutput {
                        dylib_path: cached_path,
                        module: Some(loaded),
                        shader_version: Some(version),
                        load_error: None,
                    });
                }
            }
        }

        std::fs::create_dir_all(&shader_dir).map_err(|err| {
            format!(
                "failed to create headless shader output dir `{}`: {err}",
                shader_dir.display()
            )
        })?;

        let source_path = shader_dir.join("lib.rs");
        std::fs::write(&source_path, source).map_err(|err| {
            format!(
                "failed to write generated headless shader source `{}`: {err}",
                source_path.display()
            )
        })?;

        let dylib_path = cached_path;
        // Compile to a private path and rename into place, so a crashed or
        // killed run can never leave a half-written dylib for the next one to
        // find (and so two processes building the same shader can't interleave).
        let staging_path = shader_dir.join(format!(
            "shader_{source_hash:016x}.{}.{}",
            std::process::id(),
            dylib_extension()
        ));

        let crate_name = format!("makepad_headless_shader_{source_hash:016x}");
        let output = Command::new("rustc")
            .arg("--edition=2021")
            .arg("--crate-type")
            .arg("cdylib")
            .arg("--crate-name")
            .arg(&crate_name)
            .arg("-O")
            .arg(&source_path)
            .arg("-o")
            .arg(&staging_path)
            .output()
            .map_err(|err| {
                format!("failed to run rustc for headless shader JIT `{crate_name}`: {err}")
            })?;

        if !output.status.success() {
            let _ = std::fs::remove_file(&staging_path);
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "headless shader JIT compile failed for `{}`:\n{}",
                dylib_path.display(),
                stderr.trim()
            ));
        }

        std::fs::rename(&staging_path, &dylib_path).map_err(|err| {
            format!(
                "failed to publish headless shader dylib `{}`: {err}",
                dylib_path.display()
            )
        })?;

        let mut load_error = None;
        let mut shader_version = None;
        let mut module = None;

        match HeadlessLoadedModule::load(&dylib_path) {
            Ok(loaded) => {
                shader_version = loaded.shader_version().ok();
                module = Some(loaded);
            }
            Err(err) => {
                load_error = Some(err);
            }
        }

        Ok(HeadlessJitOutput {
            dylib_path,
            module,
            shader_version,
            load_error,
        })
    }
}

fn dylib_extension() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        return "dll";
    }
    #[cfg(target_os = "macos")]
    {
        return "dylib";
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        return "so";
    }
    #[allow(unreachable_code)]
    "bin"
}

// ─────────────────────────────────────────────────────────────────────────────
// Carga dinámica del .so/.dylib de shader compilado en runtime
//
// POR QUÉ EXISTE ESTA CAPA (en español, por norma del proyecto):
//
// El backend `headless` no tiene GPU: traduce cada shader de Makepad (MPSL) a
// código Rust (`ShaderBackend::Rust`), lo compila invocando `rustc` como proceso
// hijo (ver `compile_and_load` arriba) y necesita después *ejecutar* ese código
// desde el proceso que ya está corriendo. La única forma de hacerlo sin
// reiniciar es cargar la biblioteca compartida resultante en el espacio de
// direcciones actual y resolver por nombre los símbolos que el generador emite
// (`makepad_headless_shader_version`, funciones de vértice/fragmento). Eso es
// exactamente lo que hace `dlopen`/`dlsym`: es el "linker en caliente" que
// sustituye al driver de GPU (que normalmente compilaría el shader él mismo).
//
// Originalmente sólo estaba implementado para macOS. Esta rama es común a todo
// UNIX (macOS y Linux/glibc comparten la misma API POSIX de `<dlfcn.h>` y el
// mismo valor de `RTLD_NOW`), así que el `cfg` se amplía a `unix` en lugar de
// duplicar el bloque. En glibc >= 2.34 los símbolos `dl*` viven ya dentro de
// `libc.so`, por lo que la declaración `extern "C"` de abajo resuelve en el
// enlazado sin necesidad de pedir `-ldl` explícitamente.
//
// Notas de seguridad (todo esto es `unsafe` por naturaleza):
//   - El módulo se mantiene vivo mientras exista el `HeadlessLoadedModule`; los
//     punteros a función obtenidos con `symbol()` NO deben sobrevivir al `Drop`,
//     porque `dlclose` puede desmapear el código.
//   - `RTLD_NOW` (=2 en ambos sistemas) fuerza a resolver todas las relocaciones
//     al cargar: preferimos fallar aquí, con un error legible, antes que morir
//     con un SIGSEGV en mitad del rasterizado del primer frame.
//   - `dlerror()` no es reentrante ni thread-safe; sólo se llama en el camino de
//     error, inmediatamente después de la llamada que falló.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(unix)]
pub struct HeadlessLoadedModule {
    /// Handle opaco devuelto por `dlopen`. `NonNull` porque un handle nulo es
    /// justo la señal de error de la API POSIX.
    handle: std::ptr::NonNull<c_void>,
}

#[cfg(unix)]
impl HeadlessLoadedModule {
    pub fn load(path: &Path) -> Result<Self, String> {
        const RTLD_NOW: i32 = 2;
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| format!("invalid dylib path `{}`", path.display()))?;
        let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW) };
        let handle = std::ptr::NonNull::new(handle).ok_or_else(last_dlerror)?;
        Ok(Self { handle })
    }

    pub fn shader_version(&self) -> Result<u32, String> {
        type VersionFn = unsafe extern "C" fn() -> u32;
        let version_fn: VersionFn = self.symbol("makepad_headless_shader_version")?;
        Ok(unsafe { version_fn() })
    }

    pub fn symbol<F: Sized>(&self, symbol: &str) -> Result<F, String> {
        let name = CString::new(symbol).map_err(|_| format!("invalid symbol name `{symbol}`"))?;
        let ptr = unsafe { dlsym(self.handle.as_ptr(), name.as_ptr()) };
        if ptr.is_null() {
            return Err(format!(
                "symbol `{symbol}` missing in headless shader module: {}",
                last_dlerror()
            ));
        }
        Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&ptr) })
    }
}

#[cfg(unix)]
impl Drop for HeadlessLoadedModule {
    fn drop(&mut self) {
        unsafe {
            dlclose(self.handle.as_ptr());
        }
    }
}

/// Recupera el último mensaje de error de `dlopen`/`dlsym` como `String`.
/// Devuelve un texto genérico si `dlerror()` da NULL (puede pasar si otra
/// llamada consumió el error antes).
#[cfg(unix)]
fn last_dlerror() -> String {
    let err = unsafe { dlerror() };
    if err.is_null() {
        return "unknown dlopen/dlsym error".to_string();
    }
    unsafe { CStr::from_ptr(err) }
        .to_string_lossy()
        .into_owned()
}

// Declaración manual de la API POSIX de carga dinámica. Se declara a mano (en
// lugar de depender del crate `libc`) para no añadir dependencias a
// `makepad-platform`, siguiendo el mismo estilo que el binding manual de libc
// que ya usa el backend `os/linux/libc_sys.rs`.
#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(path: *const std::os::raw::c_char, mode: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const std::os::raw::c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
    fn dlerror() -> *const std::os::raw::c_char;
}

// Fallback para plataformas sin `dlfcn.h` (Windows, wasm, y en su día ATLAS):
// el JIT de shaders no puede funcionar ahí y se reporta como error explícito en
// vez de fallar de forma silenciosa. En ATLAS la salida prevista NO es
// implementar esto, sino precompilar los shaders AOT (ver H1 del documento
// BRASA-BARE-METAL-CAMINO.md).
#[cfg(not(unix))]
pub struct HeadlessLoadedModule;

#[cfg(not(unix))]
impl HeadlessLoadedModule {
    pub fn load(path: &Path) -> Result<Self, String> {
        Err(format!(
            "headless shader dlopen is only implemented on unix for now (`{}`)",
            path.display()
        ))
    }

    pub fn shader_version(&self) -> Result<u32, String> {
        Err("headless shader version lookup is only implemented on unix for now".to_string())
    }
}
