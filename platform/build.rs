use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

/// Best-effort default for the macOS bundle name when MAKEPAD_BUNDLE_NAME isn't
/// set. Cargo doesn't expose the consuming binary's package name to a
/// dependency's build script (`CARGO_PKG_NAME` here is "makepad-platform"), so
/// we walk up from `OUT_DIR` (which is always
/// `<root>/target/<profile>/build/<crate>-<hash>/out`) to the directory that
/// contains `target/` and use that directory's name. For a typical project
/// that's the package or workspace root, which is almost always a meaningful
/// label. Capitalize the first letter so the menu bar shows "Sample app" rather
/// than "sample app". Returns `None` if the path doesn't have the expected shape
/// or the directory name isn't valid UTF-8.
fn detect_app_name(out_dir: &Path) -> Option<String> {
    let workspace_root = out_dir.ancestors().nth(5)?;
    let dir_name = workspace_root.file_name()?.to_str()?;
    if dir_name.is_empty() {
        return None;
    }
    let mut chars = dir_name.chars();
    let first = chars.next()?;
    Some(first.to_uppercase().collect::<String>() + chars.as_str())
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let path = Path::new(&out_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let cwd = std::env::current_dir().unwrap();
    let mut file = File::create(path.join("makepad-platform.path")).unwrap();
    file.write_all(format!("{}", cwd.display()).as_bytes())
        .unwrap();

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target = env::var("TARGET").unwrap();

    if target_os == "macos" {
        // The downstream app can override the bundle name shown in the macOS
        // application menu by setting MAKEPAD_BUNDLE_NAME — typically via its
        // `.cargo/config.toml` `[env]` section with `force = true`. macOS uses
        // CFBundleName from this Info.plist as the first menu bar item title
        // for unbundled `cargo run` launches, and it overrides whatever NSMenu
        // title we pass to setMainMenu:. When the env var isn't set, we fall
        // back to the workspace/package directory name (capitalized), which
        // is almost always more meaningful than a hardcoded placeholder.
        let bundle_name = env::var("MAKEPAD_BUNDLE_NAME")
            .ok()
            .or_else(|| detect_app_name(Path::new(&out_dir)))
            .unwrap_or_else(|| "Makepad App".to_string());
        let bundle_id = env::var("MAKEPAD_BUNDLE_IDENTIFIER")
            .unwrap_or_else(|_| format!("dev.makepad.{}", bundle_name.to_lowercase().replace(' ', "-")));
        let command_line_plist = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>{bundle_id}</string>
    <key>CFBundleName</key>
    <string>{bundle_name}</string>
    <key>CFBundleDisplayName</key>
    <string>{bundle_name}</string>
    <key>GCSupportsControllerUserInteraction</key>
    <true/>
    <key>GCSupportedGameControllers</key>
    <array>
        <dict>
            <key>ProfileName</key>
            <string>ExtendedGamepad</string>
        </dict>
    </array>
    <key>NSLocationUsageDescription</key>
    <string>Used to show your position on the map.</string>
    <key>NSLocationWhenInUseUsageDescription</key>
    <string>Used to show your position on the map.</string>
</dict>
</plist>
"#
        );
        std::fs::write(path.join("Info.plist"), command_line_plist).unwrap();
    }

    // Per slot: env-var override → auto-discovery in `<workspace_root>/resources/`.
    // Workspace root is the dir containing `target/` (5 ancestors up from
    // OUT_DIR), same heuristic as `detect_app_name`.
    let icons: &[(&str, &str, &str)] = &[
        ("MAKEPAD_APP_ICON_32",   "icon_32.png",   "CUSTOM_ICON_PNG_32"),
        ("MAKEPAD_APP_ICON_64",   "icon_64.png",   "CUSTOM_ICON_PNG_64"),
        ("MAKEPAD_APP_ICON_128",  "icon_128.png",  "CUSTOM_ICON_PNG_128"),
        ("MAKEPAD_APP_ICON_256",  "icon_256.png",  "CUSTOM_ICON_PNG_256"),
        ("MAKEPAD_APP_ICON_512",  "icon_512.png",  "CUSTOM_ICON_PNG_512"),
        ("MAKEPAD_APP_ICON_1024", "icon_1024.png", "CUSTOM_ICON_PNG_1024"),
        ("MAKEPAD_APP_ICON_ICO",  "icon.ico",      "CUSTOM_ICON_ICO"),
    ];
    let resources_dir = Path::new(&out_dir).ancestors().nth(5).map(|r| r.join("resources"));
    let mut icon_gen = String::new();
    for &(var, filename, const_name) in icons {
        println!("cargo:rerun-if-env-changed={var}");
        let path = env::var(var).ok().or_else(|| {
            let p = resources_dir.as_ref()?.join(filename);
            p.is_file().then(|| p.to_string_lossy().into_owned())
        });
        let value = match &path {
            Some(p) => {
                println!("cargo:rerun-if-changed={p}");
                format!("include_bytes!(r#\"{p}\"#)")
            }
            None => "&[]".to_string(),
        };
        icon_gen.push_str(&format!(
            "#[allow(dead_code)] pub static {const_name}: &'static [u8] = {value};\n"
        ));
    }
    // Watch the resources dir so new/removed icon files trigger a rebuild
    // (rerun-if-changed on a non-existent file is a no-op).
    if let Some(dir) = resources_dir.as_ref().filter(|d| d.is_dir()) {
        println!("cargo:rerun-if-changed={}", dir.display());
    }
    std::fs::write(Path::new(&out_dir).join("app_icon_gen.rs"), icon_gen).unwrap();

    generate_headless_aot_shaders(&out_dir);

    println!("cargo:rustc-check-cfg=cfg(apple_bundle,apple_sim,lines,use_gles_3,use_vulkan,linux_direct,quest,no_android_choreographer,ohos_sim,headless,use_unstable_unix_socket_ancillary_data_2021)");
    println!("cargo:rerun-if-env-changed=MAKEPAD");
    println!("cargo:rerun-if-env-changed=MAKEPAD_PACKAGE_DIR");
    println!("cargo:rerun-if-env-changed=MAKEPAD_BUNDLE_NAME");
    println!("cargo:rerun-if-env-changed=MAKEPAD_BUNDLE_IDENTIFIER");
    println!("cargo:rerun-if-env-changed=IPHONEOS_DEPLOYMENT_TARGET");

    if let Ok(configs) = env::var("MAKEPAD") {
        for config in configs.split(['+', ',']) {
            match config {
                "lines" => println!("cargo:rustc-cfg=lines"),
                "linux_direct" => println!("cargo:rustc-cfg=linux_direct"),
                "no_android_choreographer" => println!("cargo:rustc-cfg=no_android_choreographer"),
                "quest" => {
                    println!("cargo:rustc-cfg=quest");
                    println!("cargo:rustc-cfg=use_gles_3");
                    println!("cargo:rustc-cfg=use_vulkan");
                }
                "apple_bundle" => println!("cargo:rustc-cfg=apple_bundle"),
                "ohos_sim" => println!("cargo:rustc-cfg=ohos_sim"),
                "headless" => println!("cargo:rustc-cfg=headless"),
                "use_gles_3" => println!("cargo:rustc-cfg=use_gles_3"),
                "vulkan" | "use_vulkan" => println!("cargo:rustc-cfg=use_vulkan"),
                _ => {}
            }
        }
    }

    match target_os.as_str() {
        "macos" => {
            println!("cargo:rustc-link-lib=framework=GameController");
            println!("cargo:rustc-link-lib=framework=CoreLocation");
            println!("cargo:rustc-link-lib=framework=AudioToolbox");
        }
        "ios" => {
            if target == "aarch64-apple-ios-sim" {
                println!("cargo:rustc-cfg=apple_sim");
            }
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=framework=GameController");
            println!("cargo:rustc-link-lib=framework=CoreLocation");
            println!("cargo:rustc-link-lib=framework=AudioToolbox");
        }
        "tvos" => {
            if target == "aarch64-apple-tvos-sim" {
                println!("cargo:rustc-cfg=apple_sim");
            }
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=framework=GameController");
        }
        "linux" => {
            println!("cargo:rustc-cfg=use_gles_3");
            println!("cargo:rustc-link-lib=xkbcommon");
        }
        "android" => {
            println!("cargo:rustc-cfg=use_gles_3");
        }
        _ => (),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AOT de shaders del backend `headless` (hito H1 de BRASA-BARE-METAL-CAMINO.md)
//
// POR QUÉ EXISTE (en español, por norma del proyecto):
//
// El backend `headless` traduce cada shader de Makepad a código Rust y lo
// compila EN TIEMPO DE EJECUCIÓN invocando `rustc` como proceso hijo, para
// luego cargar el `.so` resultante con `dlopen` (ver `os/headless/jit.rs`).
// Eso funciona en un host de desarrollo, pero es inviable en dos escenarios
// que sí importan:
//
//   1. **ATLAS OS**: dentro del sistema operativo objetivo no hay ni compilador
//      de Rust ni cargador dinámico, así que el JIT sencillamente no puede
//      existir. Sin AOT no hay backend por software y, por tanto, no hay
//      escritorio Makepad+Brasa.
//   2. **Arranque**: medido en este repo, compilar los 50 shaders de una
//      pantalla real de Brasa cuesta ~22 segundos en el primer frame. El AOT
//      los mueve al build, donde se pagan una vez.
//
// CÓMO FUNCIONA:
//
// Se ejecuta la aplicación UNA VEZ con el JIT activo; éste deja en disco un
// `shader_<hash>/lib.rs` por shader. Ese directorio se pasa en la siguiente
// compilación mediante `MAKEPAD_HEADLESS_AOT_DIR`, y este generador empotra
// cada `lib.rs` como un módulo Rust del propio `makepad-platform`, más una
// tabla estática `hash -> resolvedor de símbolos`. En runtime,
// `HeadlessShaderJit::compile_and_load` consulta primero esa tabla y sólo cae
// al `rustc` si el shader no estaba precompilado.
//
// DECISIÓN DE DISEÑO — por qué se quita `#[no_mangle]`:
//
// Cada `lib.rs` está pensado como una `cdylib` independiente y exporta sus 11
// puntos de entrada (`makepad_headless_vertex`, `..._fragment`, los offsets
// del `RenderCx`, etc.) con `#[no_mangle]`, es decir, con nombre de símbolo
// global SIN decorar. Al empotrar 50 de esos ficheros en un mismo binario,
// esos 50×11 símbolos colisionarían y el enlazado fallaría. Se eliminan los
// `#[no_mangle]`: las funciones siguen siendo `pub extern "C"` y siguen siendo
// alcanzables por RUTA de módulo (`shader_xxxx::makepad_headless_vertex`), que
// es justo lo que la tabla generada necesita. Se deja de exportar un símbolo
// dinámico que nadie iba a buscar por nombre en este modo.
//
// La resolución por nombre se conserva (`fn(&str) -> Option<*const ()>`) para
// que el resto del backend (`raster.rs`, `shader.rs`) siga usando exactamente
// el mismo interfaz `symbol::<F>("nombre")` que usaba con `dlsym`. El AOT es
// un reemplazo transparente, no un camino paralelo.
// ─────────────────────────────────────────────────────────────────────────────

/// Los 11 puntos de entrada que el generador de shaders puede emitir. No todos
/// aparecen en todos los shaders (p. ej. `rcx_frag_offset` sólo se emite si el
/// shader lee el framebuffer), por eso la tabla se construye a partir de lo que
/// realmente hay en cada fichero y no de esta lista a ciegas.
const HEADLESS_AOT_ENTRY_POINTS: &[&str] = &[
    "makepad_headless_shader_version",
    "makepad_headless_flat_varying_slots",
    "makepad_headless_uses_derivatives",
    "makepad_headless_render_cx_size",
    "makepad_headless_rcx_vary_offset",
    "makepad_headless_rcx_quad_mode_offset",
    "makepad_headless_rcx_frag_offset",
    "makepad_headless_rcx_discard_offset",
    "makepad_headless_fill_rcx",
    "makepad_headless_vertex",
    "makepad_headless_fragment",
];

/// Genera `$OUT_DIR/headless_aot_gen.rs`.
///
/// Si `MAKEPAD_HEADLESS_AOT_DIR` no está definido (el caso normal), emite una
/// tabla VACÍA: el coste es cero y el backend usa el JIT como siempre. Esto
/// mantiene la compilación por defecto idéntica a la de upstream.
fn generate_headless_aot_shaders(out_dir: &str) {
    println!("cargo:rerun-if-env-changed=MAKEPAD_HEADLESS_AOT_DIR");
    let gen_path = Path::new(out_dir).join("headless_aot_gen.rs");

    let aot_dir = match env::var("MAKEPAD_HEADLESS_AOT_DIR") {
        Ok(dir) if !dir.trim().is_empty() => std::path::PathBuf::from(dir),
        _ => {
            std::fs::write(&gen_path, "pub static AOT_SHADERS: &[AotShader] = &[];\n").unwrap();
            return;
        }
    };
    println!("cargo:rerun-if-changed={}", aot_dir.display());

    // Directorio donde se dejan las copias despojadas de `#[no_mangle]`. Vive
    // en OUT_DIR para que `cargo clean` se lo lleve y para no ensuciar el
    // volcado original, que es la entrada y debe quedar intacta.
    let embedded_dir = Path::new(out_dir).join("headless_aot_shaders");
    std::fs::create_dir_all(&embedded_dir).unwrap();

    // Se ordenan por hash para que la tabla generada sea DETERMINISTA: dos
    // builds del mismo volcado deben producir el mismo fichero, byte a byte.
    let mut shaders: Vec<(u64, std::path::PathBuf)> = Vec::new();
    let entries = match std::fs::read_dir(&aot_dir) {
        Ok(e) => e,
        Err(err) => panic!(
            "MAKEPAD_HEADLESS_AOT_DIR `{}` no se puede leer: {err}",
            aot_dir.display()
        ),
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(hex) = name.strip_prefix("shader_") else {
            continue;
        };
        let Ok(hash) = u64::from_str_radix(hex, 16) else {
            continue;
        };
        let lib_rs = entry.path().join("lib.rs");
        if lib_rs.is_file() {
            println!("cargo:rerun-if-changed={}", lib_rs.display());
            shaders.push((hash, lib_rs));
        }
    }
    shaders.sort_by_key(|(hash, _)| *hash);

    let mut gen = String::new();
    gen.push_str("// GENERADO por platform/build.rs — no editar a mano.\n");
    let mut table = String::from("pub static AOT_SHADERS: &[AotShader] = &[\n");

    for (hash, lib_rs) in &shaders {
        let source = std::fs::read_to_string(lib_rs)
            .unwrap_or_else(|err| panic!("no se puede leer `{}`: {err}", lib_rs.display()));
        // Se busca el atributo tal cual lo emite el generador de shaders, en su
        // propia línea. Un `replace` del texto suelto podría tocar una cadena
        // literal dentro del shader; anclarlo a "línea completa" lo evita.
        let stripped: String = source
            .lines()
            .filter(|line| line.trim() != "#[no_mangle]")
            .collect::<Vec<_>>()
            .join("\n");

        let module = format!("shader_{hash:016x}");
        let dest = embedded_dir.join(format!("{module}.rs"));
        // Sólo se reescribe si cambia: evita invalidar el mtime y forzar
        // recompilaciones de 50 módulos en cada build.
        let needs_write = std::fs::read_to_string(&dest)
            .map(|old| old != stripped)
            .unwrap_or(true);
        if needs_write {
            std::fs::write(&dest, &stripped).unwrap();
        }

        gen.push_str(&format!(
            "#[path = r#\"{}\"#]\nmod {module};\n",
            dest.display()
        ));

        // Resolvedor por nombre del módulo: sustituye a `dlsym`. Sólo se
        // enumeran los puntos de entrada realmente presentes en el fichero.
        gen.push_str(&format!(
            "fn resolve_{module}(name: &str) -> Option<*const ()> {{\n    Some(match name {{\n"
        ));
        for ep in HEADLESS_AOT_ENTRY_POINTS {
            if source.contains(&format!("fn {ep}(")) {
                gen.push_str(&format!(
                    "        \"{ep}\" => {module}::{ep} as *const (),\n"
                ));
            }
        }
        gen.push_str("        _ => return None,\n    })\n}\n");

        table.push_str(&format!(
            "    AotShader {{ source_hash: 0x{hash:016x}u64, resolve: resolve_{module} }},\n"
        ));
    }
    table.push_str("];\n");
    gen.push_str(&table);

    std::fs::write(&gen_path, gen).unwrap();
    println!(
        "cargo:warning=makepad headless AOT: {} shaders empotrados desde {}",
        shaders.len(),
        aot_dir.display()
    );
}
