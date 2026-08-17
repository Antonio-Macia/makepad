use super::virtual_gpu::{
    rasterize_triangle_rows, Framebuffer, RasterScratch, TriangleDerivatives,
};
use crate::{
    cx::Cx,
    draw_list::{CxDrawKind, DrawListId},
    draw_pass::{CxDrawPassParent, DrawPassId},
    draw_shader::{CxDrawShaderCode, CxDrawShaderMapping},
    makepad_live_id::*,
    makepad_math::*,
    texture::TextureFormat,
};
use makepad_zune_png::{
    makepad_zune_core::{bit_depth::BitDepth, colorspace::ColorSpace, options::EncoderOptions},
    PngEncoder,
};
use std::{collections::HashMap, sync::mpsc};

// ─────────────────────────────────────────────────────────────────────────────
// JIT shader function pointer types
// ─────────────────────────────────────────────────────────────────────────────

type VertexFn = unsafe extern "C" fn(
    geom_ptr: *const f32,
    geom_len: u32,
    inst_ptr: *const f32,
    inst_len: u32,
    uniform_ptrs: *const *const f32,
    uniform_lens: *const u32,
    uniform_count: u32,
    varying_out: *mut f32,
    varying_len: u32,
    out_pos: *mut [f32; 4],
);

/// Fragment entry: takes a pre-filled RenderCx buffer, returns 1 = write pixel, 0 = discard.
/// The host reads frag_fb0 directly from the buffer after the call.
type FragmentFn = unsafe extern "C" fn(rcx_ptr: *mut f32, rcx_f32s: u32) -> u32;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Write a u32 value at a byte offset in the rcx buffer.
#[inline]
fn set_u32(buf: &mut [u8], offset: usize, val: u32) {
    if offset + 4 <= buf.len() {
        buf[offset..offset + 4].copy_from_slice(&val.to_ne_bytes());
    }
}

#[derive(Clone, Copy)]
struct RowChunk {
    start: usize,
    end: usize,
}

fn configured_render_threads(default_threads: usize) -> usize {
    // Efficiency-first default: avoid blasting all cores unless explicitly requested.
    let auto_threads = default_threads.min(4).max(1);
    std::env::var("MAKEPAD_HEADLESS_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(auto_threads)
}

fn configured_parallel_min_tris(default_min: usize) -> usize {
    std::env::var("MAKEPAD_HEADLESS_PARALLEL_MIN_TRIS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default_min)
}

fn compute_index_chunks(
    total: usize,
    desired_chunks: usize,
    min_items_per_chunk: usize,
) -> Vec<RowChunk> {
    if total == 0 {
        return Vec::new();
    }
    let max_chunks = (total / min_items_per_chunk.max(1)).max(1);
    let chunk_count = desired_chunks.max(1).min(max_chunks);
    if chunk_count <= 1 {
        return vec![RowChunk {
            start: 0,
            end: total,
        }];
    }

    let mut chunks = Vec::with_capacity(chunk_count);
    let base = total / chunk_count;
    let rem = total % chunk_count;
    let mut start = 0usize;
    for i in 0..chunk_count {
        let items = base + usize::from(i < rem);
        let end = (start + items).min(total);
        if end > start {
            chunks.push(RowChunk { start, end });
        }
        start = end;
    }
    if chunks.is_empty() {
        chunks.push(RowChunk {
            start: 0,
            end: total,
        });
    }
    chunks
}

fn compute_row_chunks(height: usize, desired_threads: usize) -> Vec<RowChunk> {
    compute_index_chunks(height, desired_threads, 32)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct TextureConversionSignature {
    kind: u8,
    width: usize,
    height: usize,
    data_ptr: usize,
    data_len: usize,
}

/// Diagnostic counters: how many REAL conversions (cache misses) happened and
/// how many pixels they covered. `MAKEPAD_HEADLESS_PROFILE` reads them to tell
/// "the cache is useless" apart from "the cache works but the atlas keeps
/// changing" — the two look identical in a per-frame total.
pub(crate) static TEXTURE_CONVERSIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(crate) static TEXTURE_CONVERTED_PX: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub(crate) struct CachedTextureConversion {
    signature: TextureConversionSignature,
    rgba: Vec<f32>,
}

pub(crate) type TextureConversionCache = HashMap<usize, CachedTextureConversion>;

/// Convierte (o recupera de caché) una textura al formato RGBA-f32 que espera
/// el shader generado, y devuelve `[ptr, len, ancho, alto]`.
///
/// `already_converted_this_frame`: si la textura YA se convirtió durante este
/// mismo frame, se ignora su flag `updated` y se sirve la caché. Sin esto, el
/// atlas de glifos —que llega marcado como "sucio" y sólo lo limpia la subida a
/// GPU, que aquí no existe— se reconvertía entero en cada uno de los ~48
/// draw-calls de texto de una pantalla de Brasa. El contenido no cambia a mitad
/// de frame: el atlas se rellena en la fase de `draw`, no durante el rasterizado.
fn headless_texture_info(
    texture_index: usize,
    cxtexture: &crate::texture::CxTexture,
    cache: &mut TextureConversionCache,
    already_converted_this_frame: bool,
) -> Option<[usize; 4]> {
    match &cxtexture.format {
        TextureFormat::VecMipRGBAf32 {
            width,
            height,
            data: Some(data),
            ..
        }
        | TextureFormat::VecRGBAf32 {
            width,
            height,
            data: Some(data),
            ..
        } => Some([data.as_ptr() as usize, data.len(), *width, *height]),
        TextureFormat::VecBGRAu8_32 {
            width,
            height,
            data: Some(data),
            updated,
        }
        | TextureFormat::VecMipBGRAu8_32 {
            width,
            height,
            data: Some(data),
            updated,
            ..
        } => {
            let sig = TextureConversionSignature {
                kind: 1,
                width: *width,
                height: *height,
                data_ptr: data.as_ptr() as usize,
                data_len: data.len(),
            };
            let entry = cache
                .entry(texture_index)
                .or_insert_with(|| CachedTextureConversion {
                    signature: sig,
                    rgba: Vec::new(),
                });
            if entry.signature != sig
                || (!updated.is_empty() && !already_converted_this_frame)
                || entry.rgba.is_empty()
            {
                entry.signature = sig;
                TEXTURE_CONVERSIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                TEXTURE_CONVERTED_PX
                    .fetch_add(*width * *height, std::sync::atomic::Ordering::Relaxed);
                entry.rgba.clear();
                entry.rgba.reserve(data.len() * 4);
                for &pixel in data {
                    let b = (pixel & 0xFF) as f32 / 255.0;
                    let g = ((pixel >> 8) & 0xFF) as f32 / 255.0;
                    let r = ((pixel >> 16) & 0xFF) as f32 / 255.0;
                    let a = ((pixel >> 24) & 0xFF) as f32 / 255.0;
                    entry.rgba.push(r);
                    entry.rgba.push(g);
                    entry.rgba.push(b);
                    entry.rgba.push(a);
                }
            }
            Some([
                entry.rgba.as_ptr() as usize,
                entry.rgba.len(),
                *width,
                *height,
            ])
        }
        TextureFormat::VecCubeBGRAu8_32 {
            width,
            height,
            data: Some(data),
            updated,
        } => {
            let expected = width.saturating_mul(*height).saturating_mul(6);
            let sig = TextureConversionSignature {
                kind: 4,
                width: *width,
                height: *height,
                data_ptr: data.as_ptr() as usize,
                data_len: data.len(),
            };
            let entry = cache
                .entry(texture_index)
                .or_insert_with(|| CachedTextureConversion {
                    signature: sig,
                    rgba: Vec::new(),
                });
            if entry.signature != sig
                || (!updated.is_empty() && !already_converted_this_frame)
                || entry.rgba.is_empty()
            {
                entry.signature = sig;
                TEXTURE_CONVERSIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                TEXTURE_CONVERTED_PX
                    .fetch_add(*width * *height, std::sync::atomic::Ordering::Relaxed);
                entry.rgba.clear();
                entry.rgba.reserve(expected.saturating_mul(4));
                for &pixel in data.iter().take(expected) {
                    let b = (pixel & 0xFF) as f32 / 255.0;
                    let g = ((pixel >> 8) & 0xFF) as f32 / 255.0;
                    let r = ((pixel >> 16) & 0xFF) as f32 / 255.0;
                    let a = ((pixel >> 24) & 0xFF) as f32 / 255.0;
                    entry.rgba.push(r);
                    entry.rgba.push(g);
                    entry.rgba.push(b);
                    entry.rgba.push(a);
                }
            }
            Some([
                entry.rgba.as_ptr() as usize,
                entry.rgba.len(),
                *width,
                *height,
            ])
        }
        TextureFormat::VecRu8 {
            width,
            height,
            data: Some(data),
            updated,
            ..
        } => {
            let expected = width.saturating_mul(*height);
            let sig = TextureConversionSignature {
                kind: 2,
                width: *width,
                height: *height,
                data_ptr: data.as_ptr() as usize,
                data_len: data.len(),
            };
            let entry = cache
                .entry(texture_index)
                .or_insert_with(|| CachedTextureConversion {
                    signature: sig,
                    rgba: Vec::new(),
                });
            if entry.signature != sig
                || (!updated.is_empty() && !already_converted_this_frame)
                || entry.rgba.is_empty()
            {
                entry.signature = sig;
                TEXTURE_CONVERSIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                TEXTURE_CONVERTED_PX
                    .fetch_add(*width * *height, std::sync::atomic::Ordering::Relaxed);
                entry.rgba.clear();
                entry.rgba.reserve(expected * 4);
                for &byte in data.iter().take(expected) {
                    let v = byte as f32 / 255.0;
                    entry.rgba.push(v);
                    entry.rgba.push(v);
                    entry.rgba.push(v);
                    entry.rgba.push(v);
                }
            }
            Some([
                entry.rgba.as_ptr() as usize,
                entry.rgba.len(),
                *width,
                *height,
            ])
        }
        TextureFormat::VecRf32 {
            width,
            height,
            data: Some(data),
            updated,
        } => {
            let expected = width.saturating_mul(*height);
            let sig = TextureConversionSignature {
                kind: 3,
                width: *width,
                height: *height,
                data_ptr: data.as_ptr() as usize,
                data_len: data.len(),
            };
            let entry = cache
                .entry(texture_index)
                .or_insert_with(|| CachedTextureConversion {
                    signature: sig,
                    rgba: Vec::new(),
                });
            if entry.signature != sig
                || (!updated.is_empty() && !already_converted_this_frame)
                || entry.rgba.is_empty()
            {
                entry.signature = sig;
                TEXTURE_CONVERSIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                TEXTURE_CONVERTED_PX
                    .fetch_add(*width * *height, std::sync::atomic::Ordering::Relaxed);
                entry.rgba.clear();
                entry.rgba.reserve(expected * 4);
                for &v in data.iter().take(expected) {
                    entry.rgba.push(v);
                    entry.rgba.push(v);
                    entry.rgba.push(v);
                    entry.rgba.push(v);
                }
            }
            Some([
                entry.rgba.as_ptr() as usize,
                entry.rgba.len(),
                *width,
                *height,
            ])
        }
        _ => None,
    }
}

#[derive(Default)]
struct RenderProfile {
    draw_calls: usize,
    parallel_draw_calls: usize,
    serial_draw_calls: usize,
    total_instances: usize,
    total_triangles: usize,
    vertex_ms: f64,
    raster_ms: f64,
    texture_ms: f64,
    /// Coste de reservar y limpiar el framebuffer, aparte del rasterizado.
    /// Separado a proposito: en una ventana grande puede dominar el frame, y
    /// sumado a `raster_ms` haria pensar que lo caro es dibujar.
    framebuffer_ms: f64,
    /// Tiempo de PREPARACIÓN por draw-call, sin rasterizar: resolver el shader,
    /// montar los buffers de uniforms, rellenar la plantilla de `RenderCx`.
    /// Es la parte del coste que NO depende del área pintada, y por tanto la que
    /// decide si el repintado parcial sirve de algo (H0-bis, ATLAS).
    setup_ms: f64,
    /// Desglose por shader: `debug_id -> (ms de raster, fragmentos sombreados,
    /// nº de draw-calls)`. Sirve para saber qué shader se come el frame.
    per_shader: HashMap<String, (f64, u64, usize)>,
}

/// Un draw-call ya PREPARADO, listo para rasterizarse sobre cualquier franja de
/// filas. Contiene copias propias de todo lo que necesita el rasterizador, así
/// que varios hilos pueden ejecutarlo en paralelo sobre franjas disjuntas sin
/// tocar el `Cx` (que no es `Send`).
///
/// Los punteros a función del shader JIT son `extern "C" fn`, que sí son
/// `Send`/`Sync`; el módulo `.so` que los contiene lo mantiene vivo el `Cx`
/// durante todo el frame.
struct BandJob {
    indices: Vec<u32>,
    instance_count: usize,
    vertex_count: usize,
    varying_slots: usize,
    shaded_positions: Vec<[f32; 4]>,
    shaded_varyings: Vec<f32>,
    flat_slots: usize,
    rcx_template: Vec<u8>,
    rcx_size: usize,
    rcx_f32s: usize,
    rcx_vary_offset: usize,
    rcx_quad_mode_offset: usize,
    rcx_frag_offset: usize,
    uses_derivatives: bool,
    fragment_fn: FragmentFn,
    is_draw_text_shader: bool,
}

/// Rasteriza el plan completo con `bands` hilos, uno por franja horizontal.
///
/// El framebuffer se parte en trozos de filas DISJUNTOS con `split_at_mut`, así
/// que no hay `unsafe`, no hay contención de escritura y el orden de pintado se
/// conserva dentro de cada franja (que es lo único que importa: dos franjas
/// nunca escriben el mismo píxel).
fn run_band_jobs(fb: &mut Framebuffer, jobs: &[BandJob], bands: usize) {
    let width = fb.width;
    let height = fb.height;
    let chunks = compute_index_chunks(height, bands, 1);

    // Trocear color y depth en rebanadas de filas alineadas con `chunks`.
    let mut color_rest: &mut [[f32; 4]] = fb.color.as_mut_slice();
    let mut depth_rest: &mut [f32] = fb.depth.as_mut_slice();
    let mut parts: Vec<(usize, usize, &mut [[f32; 4]], &mut [f32])> = Vec::new();
    let mut consumed = 0usize;
    for c in &chunks {
        let rows = c.end - c.start;
        let px = rows * width;
        let (cl, cr) = color_rest.split_at_mut(px);
        let (dl, dr) = depth_rest.split_at_mut(px);
        color_rest = cr;
        depth_rest = dr;
        parts.push((c.start, c.end, cl, dl));
        consumed += rows;
    }
    debug_assert_eq!(consumed, height);

    std::thread::scope(|scope| {
        for (row_start, row_end, color_chunk, depth_chunk) in parts {
            scope.spawn(move || {
                for j in jobs {
                    rasterize_instances_rows(
                        color_chunk,
                        depth_chunk,
                        width,
                        height,
                        row_start,
                        row_end,
                        &j.indices,
                        j.instance_count,
                        j.vertex_count,
                        j.varying_slots,
                        &j.shaded_positions,
                        &j.shaded_varyings,
                        j.flat_slots,
                        &j.rcx_template,
                        j.rcx_size,
                        j.rcx_f32s,
                        j.rcx_vary_offset,
                        j.rcx_quad_mode_offset,
                        j.rcx_frag_offset,
                        j.uses_derivatives,
                        j.fragment_fn,
                        false,
                        j.is_draw_text_shader,
                    );
                }
            });
        }
    });
}

#[allow(clippy::too_many_arguments)]
/// Cobertura ya calculada de UNA instancia (un glifo) dentro de su cuadrado.
///
/// 🔴 POR QUÉ EXISTE. El shader de texto evalúa las curvas del glifo **por
/// píxel**, y con derivadas se invoca **tres veces** por fragmento (dx, dy y
/// centro). Medido en ATLAS/H0: ~11.500 ns/fragmento frente a ~30 ns de los
/// `sdf` — con el 1 % de los fragmentos se comía el 57 % del rasterizado.
///
/// Es la técnica clásica de los motores 2D desde los noventa: **rasterizar el
/// glifo UNA vez y luego copiarlo**. Un `memcpy` mueve un píxel en ~0,2 ns; aquí
/// se calculaba uno en 11.500.
///
/// La clave garantiza la corrección POR CONSTRUCCIÓN: entra el hash de TODOS los
/// varyings de los vértices de la instancia más el tamaño entero del cuadrado y
/// su desplazamiento subpíxel. Dos instancias con la misma clave producen, en el
/// mismo píxel local, exactamente el mismo resultado — no hay que saber qué slot
/// guarda qué.
struct GlyphTile {
    w: usize,
    h: usize,
    /// `None` = ese píxel local aún no se ha calculado. Se llena perezosamente
    /// porque las bandas reparten FILAS: cada banda calcula sólo las suyas.
    px: Vec<Option<[f32; 4]>>,
}

/// Clave estable de una instancia de texto. Ver [`GlyphTile`].
fn glyph_key(
    varyings: &[f32],
    v_off: usize,
    varying_slots: usize,
    vertex_count: usize,
    sub_x: f32,
    sub_y: f32,
    w: usize,
    h: usize,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for v in 0..vertex_count {
        let base = v_off + v * varying_slots;
        if base + varying_slots > varyings.len() {
            return 0;
        }
        for f in &varyings[base..base + varying_slots] {
            f.to_bits().hash(&mut hasher);
        }
    }
    // Subpíxel cuantizado a 1/8: un glifo en x=100,3 no cubre igual que en
    // x=100,7, así que si no entrara en la clave la caché mentiría.
    ((sub_x * 8.0).round() as i32).hash(&mut hasher);
    ((sub_y * 8.0).round() as i32).hash(&mut hasher);
    w.hash(&mut hasher);
    h.hash(&mut hasher);
    hasher.finish()
}

/// Sonda: **saltarse del todo** las dos pasadas de grabación
/// (`MAKEPAD_HEADLESS_RECORD_NOOP`).
///
/// ⚠ **NO es una optimización: rompe las derivadas** (los quad buffers se quedan
/// sin escribir, así que `dFdx`/`dFdy` devuelven basura del píxel anterior). Es
/// una **sonda de diagnóstico**, hermana de `MAKEPAD_HEADLESS_NO_DERIV`, y sirve
/// para separar dos cosas que esa otra mezcla:
///
/// - `NO_DERIV` apaga el camino entero, así que ahorra **el shader Y el
///   andamiaje del host** (rellenar `dx_varyings`/`dy_varyings` sobre todos los
///   slots y tres `write_varyings` por píxel).
/// - Ésta ahorra **sólo las dos ejecuciones del shader**, dejando intacto todo
///   el trabajo del host.
///
/// La diferencia entre las dos dice dónde está el coste de verdad, que es
/// justamente lo que hacía falta saber para decidir si merece la pena recortar
/// el shader o hay que atacar el host.
fn record_noop() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MAKEPAD_HEADLESS_RECORD_NOOP").is_ok())
}

/// ¿Está encendido el memo de glifos? (`MAKEPAD_HEADLESS_GLYPH_MEMO`)
///
/// Apagado por defecto: hoy no acierta (ver `GlyphTile`) y encenderlo sólo añade
/// trabajo. Existe para que el siguiente que lo retome mida en vez de suponer.
fn glyph_memo_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MAKEPAD_HEADLESS_GLYPH_MEMO").is_ok())
}

/// Contadores del memo de glifos (diagnóstico, se reinician por frame).
pub(crate) static GLYPH_MEMO_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub(crate) static GLYPH_MEMO_MISS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn rasterize_instances_rows(
    color_chunk: &mut [[f32; 4]],
    depth_chunk: &mut [f32],
    width: usize,
    height: usize,
    row_start: usize,
    row_end: usize,
    indices: &[u32],
    instance_count: usize,
    vertex_count: usize,
    varying_slots: usize,
    shaded_positions: &[[f32; 4]],
    shaded_varyings: &[f32],
    flat_slots: usize,
    rcx_template: &[u8],
    rcx_size: usize,
    rcx_f32s: usize,
    rcx_vary_offset: usize,
    rcx_quad_mode_offset: usize,
    rcx_frag_offset: usize,
    uses_derivatives: bool,
    fragment_fn: FragmentFn,
    debug_text: bool,
    is_draw_text_shader: bool,
) {
    let mut rcx_buf = rcx_template.to_vec();
    let mut dx_varyings = if uses_derivatives {
        vec![0.0f32; varying_slots]
    } else {
        Vec::new()
    };
    let mut dy_varyings = if uses_derivatives {
        vec![0.0f32; varying_slots]
    } else {
        Vec::new()
    };
    let shift_start = flat_slots.min(varying_slots);
    // Sonda de diagnóstico: ver `record_noop`. Rompe las derivadas a propósito.
    let saltar_grabacion = record_noop();
    let tri_count = indices.len() / 3;
    let vary_bytes = varying_slots * std::mem::size_of::<f32>();
    let mut debug_text_prints = 0usize;
    let mut raster_scratch = RasterScratch::default();

    // Memo de glifos: vive por LLAMADA, o sea por banda y por hilo. No hace falta
    // candado ni `thread_local`, y encaja con el reparto por filas: el texto de
    // una línea cae en la misma banda, que es justo donde se repiten las letras.
    let mut glyph_memo: std::collections::HashMap<u64, GlyphTile> =
        std::collections::HashMap::new();

    for inst_idx in 0..instance_count {
        let inst_base = inst_idx * vertex_count;

        // Caja envolvente de la INSTANCIA (no del triángulo): un glifo son dos
        // triángulos y la caché se indexa por su cuadrado completo.
        let mut tile: Option<(u64, i32, i32, usize, usize)> = None;
        // 🔴 APAGADO POR DEFECTO — experimento incompleto, ver `GlyphTile`.
        // Medido el 2026-08-16: se activa bien (cajas de 7×10 px, glifos reales)
        // pero da **0 aciertos de 23.384**, porque la clave incluye TODOS los
        // varyings del vértice y esos llevan la POSICIÓN EN PANTALLA: la misma
        // letra en dos sitios distintos son claves distintas. Encendido sólo
        // cuesta (un hash por instancia y una tabla que nunca acierta), así que
        // se deja tras una variable hasta que la clave sepa distinguir la
        // identidad del glifo de su posición.
        if is_draw_text_shader && glyph_memo_enabled() {
            let (mut mnx, mut mny, mut mxx, mut mxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            let mut ok = true;
            for v in 0..vertex_count {
                match shaded_positions.get(inst_base + v) {
                    Some(p) => {
                        // 🔴 Las posiciones vienen en espacio de RECORTE (-1..1),
                        // no en píxeles. Sin esta conversión la caja salía de 1×1
                        // en el origen (-1,-1) y la comprobación de límites
                        // fallaba SIEMPRE: el memo no llegaba a activarse ni una
                        // vez, y ni siquiera contaba el fallo. Misma fórmula que
                        // `ndc_to_screen` de `virtual_gpu.rs`, incluido el volteo
                        // de Y y la división por w.
                        let inv_w = if p[3] != 0.0 { 1.0 / p[3] } else { 1.0 };
                        let sx = ((p[0] * inv_w) * 0.5 + 0.5) * width as f32;
                        let sy = (1.0 - ((p[1] * inv_w) * 0.5 + 0.5)) * height as f32;
                        mnx = mnx.min(sx);
                        mny = mny.min(sy);
                        mxx = mxx.max(sx);
                        mxy = mxy.max(sy);
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && mxx > mnx && mxy > mny {
                let x0 = mnx.floor() as i32;
                let y0 = mny.floor() as i32;
                let w = ((mxx.ceil() as i32) - x0).max(1) as usize;
                let h = ((mxy.ceil() as i32) - y0).max(1) as usize;
                // Glifos desmesurados no se cachean: serían baldosas enormes con
                // pocas repeticiones, o sea memoria a cambio de nada.
                if std::env::var("MAKEPAD_HEADLESS_GLYPH_DEBUG").is_ok() {
                    static UNA: std::sync::atomic::AtomicUsize =
                        std::sync::atomic::AtomicUsize::new(0);
                    if UNA.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 5 {
                        eprintln!("[glifo] bbox w={w} h={h} x0={x0} y0={y0} verts={vertex_count} slots={varying_slots}");
                    }
                }
                if w <= 256 && h <= 256 {
                    let key = glyph_key(
                        shaded_varyings,
                        inst_base * varying_slots,
                        varying_slots,
                        vertex_count,
                        mnx - x0 as f32,
                        mny - y0 as f32,
                        w,
                        h,
                    );
                    if key != 0 {
                        tile = Some((key, x0, y0, w, h));
                    }
                }
            }
        }
        for tri_idx in 0..tri_count {
            let i0 = indices[tri_idx * 3] as usize;
            let i1 = indices[tri_idx * 3 + 1] as usize;
            let i2 = indices[tri_idx * 3 + 2] as usize;

            if i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count {
                continue;
            }

            let v0_idx = inst_base + i0;
            let v1_idx = inst_base + i1;
            let v2_idx = inst_base + i2;

            if v0_idx >= shaded_positions.len()
                || v1_idx >= shaded_positions.len()
                || v2_idx >= shaded_positions.len()
            {
                continue;
            }

            let v0_off = v0_idx * varying_slots;
            let v1_off = v1_idx * varying_slots;
            let v2_off = v2_idx * varying_slots;

            if v0_off + varying_slots > shaded_varyings.len()
                || v1_off + varying_slots > shaded_varyings.len()
                || v2_off + varying_slots > shaded_varyings.len()
            {
                continue;
            }

            let p0 = &shaded_positions[v0_idx];
            let p1 = &shaded_positions[v1_idx];
            let p2 = &shaded_positions[v2_idx];
            let vary0 = &shaded_varyings[v0_off..v0_off + varying_slots];
            let vary1 = &shaded_varyings[v1_off..v1_off + varying_slots];
            let vary2 = &shaded_varyings[v2_off..v2_off + varying_slots];

            if uses_derivatives {
                let mut frag_closure = |varyings: &[f32],
                                        derivs: &TriangleDerivatives,
                                        lane_x: u32,
                                        lane_y: u32,
                                        x: i32,
                                        y: i32|
                 -> Option<[f32; 4]> {
                    for i in 0..varyings.len() {
                        if i < shift_start {
                            dx_varyings[i] = varyings[i];
                            dy_varyings[i] = varyings[i];
                        } else {
                            dx_varyings[i] = varyings[i] + derivs.dvary_dx[i];
                            dy_varyings[i] = varyings[i] + derivs.dvary_dy[i];
                        }
                    }

                    set_u32(&mut rcx_buf, rcx_quad_mode_offset + 8, lane_x);
                    set_u32(&mut rcx_buf, rcx_quad_mode_offset + 12, lane_y);
                    write_varyings(
                        &mut rcx_buf,
                        rcx_vary_offset,
                        &dx_varyings,
                        vary_bytes,
                        rcx_size,
                    );
                    set_u32(&mut rcx_buf, rcx_quad_mode_offset, 0);
                    set_u32(&mut rcx_buf, rcx_quad_mode_offset + 4, 0);
                    if !saltar_grabacion {
                        unsafe {
                            fragment_fn(rcx_buf.as_mut_ptr() as *mut f32, rcx_f32s as u32);
                        }
                    }

                    write_varyings(
                        &mut rcx_buf,
                        rcx_vary_offset,
                        &dy_varyings,
                        vary_bytes,
                        rcx_size,
                    );
                    set_u32(&mut rcx_buf, rcx_quad_mode_offset, 1);
                    set_u32(&mut rcx_buf, rcx_quad_mode_offset + 4, 0);
                    if !saltar_grabacion {
                        unsafe {
                            fragment_fn(rcx_buf.as_mut_ptr() as *mut f32, rcx_f32s as u32);
                        }
                    }

                    write_varyings(
                        &mut rcx_buf,
                        rcx_vary_offset,
                        varyings,
                        vary_bytes,
                        rcx_size,
                    );
                    set_u32(&mut rcx_buf, rcx_quad_mode_offset, 2);
                    set_u32(&mut rcx_buf, rcx_quad_mode_offset + 4, 0);
                    let write_pixel =
                        unsafe { fragment_fn(rcx_buf.as_mut_ptr() as *mut f32, rcx_f32s as u32) };
                    if write_pixel == 0 {
                        return None;
                    }

                    if rcx_frag_offset + 16 <= rcx_size {
                        let color_ptr =
                            unsafe { rcx_buf.as_ptr().add(rcx_frag_offset) as *const [f32; 4] };
                        let color = unsafe { *color_ptr };
                        if debug_text && is_draw_text_shader && debug_text_prints < 120 {
                            let text_t_slot = shift_start + 2;
                            if text_t_slot + 1 < varyings.len() {
                                let a = color[3];
                                if a > 0.0 && a < 1.0 {
                                    eprintln!(
                                        "[headless][draw_text] px=({}, {}) lane=({}, {}) t=({:.6}, {:.6}) dFdx(t)=({:.6}, {:.6}) dFdy(t)=({:.6}, {:.6}) a={:.5}",
                                        x,
                                        y,
                                        lane_x,
                                        lane_y,
                                        varyings[text_t_slot],
                                        varyings[text_t_slot + 1],
                                        derivs.dvary_dx[text_t_slot],
                                        derivs.dvary_dx[text_t_slot + 1],
                                        derivs.dvary_dy[text_t_slot],
                                        derivs.dvary_dy[text_t_slot + 1],
                                        a,
                                    );
                                    debug_text_prints += 1;
                                }
                            }
                        }
                        Some(color)
                    } else {
                        Some([0.0, 0.0, 0.0, 0.0])
                    }
                };

                // ── Memo de glifos ────────────────────────────────────────
                // Envuelve al shader: si este píxel local de ESTE glifo ya se
                // calculó, se copia; si no, se calcula una vez y se guarda.
                // Correcto por construcción (ver `GlyphTile`): misma clave +
                // mismo píxel local ⇒ mismo resultado.
                let memo_ref = &mut glyph_memo;
                let mut frag_closure = |varyings: &[f32],
                                        derivs: &TriangleDerivatives,
                                        lane_x: u32,
                                        lane_y: u32,
                                        x: i32,
                                        y: i32|
                 -> Option<[f32; 4]> {
                    let Some((key, x0, y0, tw, th)) = tile else {
                        return frag_closure(varyings, derivs, lane_x, lane_y, x, y);
                    };
                    let (lx, ly) = (x - x0, y - y0);
                    if lx < 0 || ly < 0 || lx as usize >= tw || ly as usize >= th {
                        return frag_closure(varyings, derivs, lane_x, lane_y, x, y);
                    }
                    let idx = ly as usize * tw + lx as usize;
                    if let Some(t) = memo_ref.get(&key) {
                        if let Some(Some(c)) = t.px.get(idx) {
                            GLYPH_MEMO_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            return Some(*c);
                        }
                    }
                    GLYPH_MEMO_MISS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let out = frag_closure(varyings, derivs, lane_x, lane_y, x, y);
                    if let Some(c) = out {
                        let t = memo_ref.entry(key).or_insert_with(|| GlyphTile {
                            w: tw,
                            h: th,
                            px: vec![None; tw * th],
                        });
                        if t.w == tw && t.h == th {
                            t.px[idx] = Some(c);
                        }
                    }
                    out
                };

                rasterize_triangle_rows(
                    width,
                    height,
                    row_start,
                    row_end,
                    color_chunk,
                    depth_chunk,
                    p0,
                    vary0,
                    p1,
                    vary1,
                    p2,
                    vary2,
                    flat_slots,
                    true,
                    &mut raster_scratch,
                    &mut frag_closure,
                );
            } else {
                let mut frag_closure = |varyings: &[f32],
                                        _derivs: &TriangleDerivatives,
                                        lane_x: u32,
                                        lane_y: u32,
                                        _x: i32,
                                        _y: i32|
                 -> Option<[f32; 4]> {
                    set_u32(&mut rcx_buf, rcx_quad_mode_offset + 8, lane_x);
                    set_u32(&mut rcx_buf, rcx_quad_mode_offset + 12, lane_y);
                    write_varyings(
                        &mut rcx_buf,
                        rcx_vary_offset,
                        varyings,
                        vary_bytes,
                        rcx_size,
                    );
                    set_u32(&mut rcx_buf, rcx_quad_mode_offset, 2);
                    set_u32(&mut rcx_buf, rcx_quad_mode_offset + 4, 0);
                    let write_pixel =
                        unsafe { fragment_fn(rcx_buf.as_mut_ptr() as *mut f32, rcx_f32s as u32) };
                    if write_pixel == 0 {
                        return None;
                    }
                    if rcx_frag_offset + 16 <= rcx_size {
                        let color_ptr =
                            unsafe { rcx_buf.as_ptr().add(rcx_frag_offset) as *const [f32; 4] };
                        Some(unsafe { *color_ptr })
                    } else {
                        Some([0.0, 0.0, 0.0, 0.0])
                    }
                };

                rasterize_triangle_rows(
                    width,
                    height,
                    row_start,
                    row_end,
                    color_chunk,
                    depth_chunk,
                    p0,
                    vary0,
                    p1,
                    vary1,
                    p2,
                    vary2,
                    flat_slots,
                    false,
                    &mut raster_scratch,
                    &mut frag_closure,
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

impl Cx {
    fn headless_render_thread_count(&self) -> usize {
        let cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(self.cpu_cores.max(1));
        configured_render_threads(cpu_threads.max(1))
    }

    fn headless_ensure_render_pool(&mut self, threads: usize) {
        let threads = threads.max(1);
        if threads <= 1 {
            return;
        }
        if self.os.render_pool.is_none() || self.os.render_pool_threads != threads {
            self.os.render_pool = Some(crate::thread::MessageThreadPool::new(self, threads));
            self.os.render_pool_threads = threads;
        }
    }

    /// Render all dirty passes into the PERSISTENT window framebuffers.
    ///
    /// Returns the ids of the windows that were rendered. The framebuffers
    /// themselves stay in `self.os.framebuffers`; take them with `mem::take` and
    /// put them back when done (see `headless_emit_frames`). They are not returned
    /// by value because they must survive the frame: with partial repaint, what is
    /// outside the damage rect is *last frame's* pixels, and handing ownership out
    /// would either lose them or force a multi-megabyte copy per frame.
    pub(crate) fn headless_render_all_passes(&mut self, time: f64) -> Vec<usize> {
        let frame_start = std::time::Instant::now();
        let profile_enabled = std::env::var("MAKEPAD_HEADLESS_PROFILE").is_ok();
        let parallel_min_tris = configured_parallel_min_tris(1);
        let mut profile = RenderProfile::default();
        let mut passes_todo = Vec::new();
        self.compute_pass_repaint_order(&mut passes_todo);
        let render_threads = self.headless_render_thread_count();
        self.headless_ensure_render_pool(render_threads);

        let mut results = Vec::new();
        // Los framebuffers salen del almacén persistente y vuelven al final. Se
        // sacan (no se toman prestados) porque el bucle de abajo necesita `&mut
        // self` para el resto del render.
        let mut stored = std::mem::take(&mut self.os.framebuffers);
        let mut texture_cache = std::mem::take(&mut self.os.texture_conversions);
        // Texturas ya convertidas EN ESTE frame. Una textura puede usarse en
        // varias draw calls; sin esto, el atlas de glifos se reconvierte una vez
        // por llamada en vez de una vez por frame.
        let mut converted_textures: Vec<crate::texture::TextureId> = Vec::new();

        for draw_pass_id in &passes_todo {
            self.passes[*draw_pass_id].paint_dirty = false;

            let parent = self.passes[*draw_pass_id].parent.clone();
            match parent {
                CxDrawPassParent::Window(window_id) => {
                    let window = &self.windows[window_id];
                    let size = window.window_geom.inner_size;
                    let dpi_factor = window.window_geom.dpi_factor;

                    let width = (size.x * dpi_factor).round().max(1.0) as usize;
                    let height = (size.y * dpi_factor).round().max(1.0) as usize;

                    // Set up pass uniforms
                    if !self.passes[*draw_pass_id].keep_camera_matrix {
                        self.passes[*draw_pass_id].set_ortho_matrix(dvec2(0.0, 0.0), size);
                    }
                    self.passes[*draw_pass_id].set_dpi_factor(dpi_factor);
                    self.passes[*draw_pass_id].set_time(time as f32);

                    let fb_start = std::time::Instant::now();

                    // ── Framebuffer PERSISTENTE ───────────────────────────────
                    // Se reutiliza el del frame anterior si el tamaño coincide.
                    // Ese "si" es toda la corrección: cuando NO coincide (primer
                    // frame, redimensión) no hay pixeles anteriores que conservar,
                    // así que el recorte por daño no aplica y hay que limpiar
                    // entero. Tratar los dos casos igual dejaría la primera
                    // pantalla con basura fuera del rectángulo sucio.
                    let idx = window_id.id();
                    if stored.len() <= idx {
                        stored.resize_with(idx + 1, || None);
                    }
                    let reutilizable = matches!(
                        &stored[idx],
                        Some(f) if f.width == width && f.height == height
                    );
                    let mut fb = match stored[idx].take() {
                        Some(f) if reutilizable => f,
                        _ => Framebuffer::new(width, height),
                    };

                    // 🔴 Y lo que de verdad hace correcto el repintado parcial: el
                    // daño se SUSPENDE mientras no haya frame anterior. Afecta al
                    // rasterizado y al present, no sólo al borrado — que era el
                    // agujero de la primera versión de esto.
                    super::virtual_gpu::set_clip_suspended(!reutilizable);

                    // Daño calculado del árbol de dibujo. Se publica ANTES de
                    // limpiar, porque el borrado ya lo consulta.
                    if super::damage::damage_enabled() {
                        // El tracker se saca del `Cx` para poder pasarle el `Cx`
                        // en inmutable: necesita leer el árbol de draw lists, y
                        // vive dentro de ese mismo `Cx`.
                        let mut tracker = std::mem::take(&mut self.os.damage);
                        let d = tracker.calcular(self, dpi_factor, width as i32, height as i32);
                        self.os.damage = tracker;
                        super::virtual_gpu::set_damage_rect(d.map(|r| (r.x0, r.y0, r.x1, r.y1)));
                        if profile_enabled {
                            match d {
                                Some(r) => crate::log!(
                                    "[headless][profile] daño={}x{} en ({},{}) = {:.1}% de la pantalla",
                                    r.x1 - r.x0,
                                    r.y1 - r.y0,
                                    r.x0,
                                    r.y0,
                                    100.0 * r.area() as f64 / (width * height) as f64
                                ),
                                None => crate::log!(
                                    "[headless][profile] daño=PANTALLA ENTERA"
                                ),
                            }
                        }
                    }

                    let clear = self.passes[*draw_pass_id].clear_color;
                    let clear_rgba = [clear.x, clear.y, clear.z, clear.w];
                    match super::virtual_gpu::headless_clip_rect() {
                        // Repintado parcial de verdad: se limpia y se rasteriza
                        // sólo el daño, y lo de fuera son los pixeles del frame
                        // anterior, que siguen ahí.
                        Some((cx0, cy0, cx1, cy1)) => {
                            fb.clear_rect(clear_rgba, 1.0, cx0, cy0, cx1, cy1);
                        }
                        // Sin daño declarado, o framebuffer nuevo: pantalla entera.
                        None => fb.clear(clear_rgba, 1.0),
                    }
                    profile.framebuffer_ms += fb_start.elapsed().as_secs_f64() * 1000.0;

                    self.headless_draw_pass(
                        *draw_pass_id,
                        render_threads,
                        parallel_min_tris,
                        &mut fb,
                        &mut texture_cache,
                        &mut converted_textures,
                        if profile_enabled {
                            Some(&mut profile)
                        } else {
                            None
                        },
                    );
                    stored[idx] = Some(fb);
                    results.push(idx);
                }
                CxDrawPassParent::DrawPass(_dep_pass_id) => {
                    // TODO: render-to-texture passes
                }
                _ => {}
            }
        }

        // Consumir la marca de "sucio" de las texturas que se han convertido en
        // este frame.
        //
        // 🔴 ESTE ERA EL DEFECTO, y costaba mas de la mitad del frame en un
        // repintado por dano. TODOS los backends reales llaman a `take_updated()`
        // al subir la textura a la GPU (opengl.rs:2372, vulkan.rs:5271,
        // d3d11.rs:1499, metal.rs:2014, web_gl.rs:95); el backend por software
        // NO lo hacia. Resultado: la textura quedaba marcada como pendiente PARA
        // SIEMPRE, la condicion `!updated.is_empty()` se cumplia en cada frame, y
        // la cache de conversion NO PODIA ACERTAR NUNCA -- reconvertia el atlas de
        // glifos entero (2048x2048 = 4,19 M pixeles) en cada frame.
        //
        // Medido antes del arreglo: `conversiones=1 px_convertidos=4194304` en los
        // 8 frames de una corrida, tanto a pantalla completa como con recorte. Los
        // contadores se reinician por frame (`swap(0)`), asi que eso era una
        // conversion COMPLETA por frame, no una en total.
        //
        // Se hace aqui, al cerrar el frame, y no en el sitio de la conversion,
        // porque alli la textura se tiene prestada en INMUTABLE. `converted_textures`
        // ya venia recorriendo el arbol de dibujo con la lista exacta: solo faltaba
        // usarla.
        for texture_id in converted_textures {
            self.textures[texture_id].take_updated();
        }

        // Devolver los framebuffers al almacén: son los pixeles que el frame que
        // viene conservará fuera de su daño.
        self.os.framebuffers = stored;

        // Hand the conversions back for the next frame to reuse.
        self.os.texture_conversions = texture_cache;

        let elapsed = frame_start.elapsed();
        if profile_enabled {
            crate::log!(
                "[headless] frame render: {:.1}ms",
                elapsed.as_secs_f64() * 1000.0
            );
        }
        if profile_enabled {
            crate::log!(
                "[headless][profile] draws={} serial={} parallel={} inst={} tris={} vertex={:.1}ms raster={:.1}ms texture={:.1}ms",
                profile.draw_calls,
                profile.serial_draw_calls,
                profile.parallel_draw_calls,
                profile.total_instances,
                profile.total_triangles,
                profile.vertex_ms,
                profile.raster_ms,
                profile.texture_ms
            );
            let gm_hit = GLYPH_MEMO_HITS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let gm_miss = GLYPH_MEMO_MISS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let gm_total = gm_hit + gm_miss;
            crate::log!(
                "[headless][profile] glifos: aciertos={} fallos={} tasa={:.1}%",
                gm_hit,
                gm_miss,
                if gm_total > 0 { 100.0 * gm_hit as f64 / gm_total as f64 } else { 0.0 }
            );
            let convs = TEXTURE_CONVERSIONS.swap(0, std::sync::atomic::Ordering::Relaxed);
            let conv_px = TEXTURE_CONVERTED_PX.swap(0, std::sync::atomic::Ordering::Relaxed);
            crate::log!(
                "[headless][profile] texture={:.1}ms framebuffer={:.1}ms setup={:.1}ms conversiones={} px_convertidos={}",
                profile.texture_ms,
                profile.framebuffer_ms,
                profile.setup_ms,
                convs,
                conv_px
            );
            let tested = super::virtual_gpu::FRAG_TESTED
                .swap(0, std::sync::atomic::Ordering::Relaxed);
            let shaded = super::virtual_gpu::FRAG_SHADED
                .swap(0, std::sync::atomic::Ordering::Relaxed);
            crate::log!(
                "[headless][profile] frag_tested={} frag_shaded={} ns_por_frag={:.1}",
                tested,
                shaded,
                if shaded > 0 {
                    profile.raster_ms * 1.0e6 / shaded as f64
                } else {
                    0.0
                }
            );
            let mut rows: Vec<(String, (f64, u64, usize))> =
                profile.per_shader.iter().map(|(k, v)| (k.clone(), *v)).collect();
            rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            for (name, (ms, frags, calls)) in rows.iter().take(12) {
                crate::log!(
                    "[headless][shader] {:<28} {:>7.1}ms  frags={:<9} draws={:<3} ns/frag={:.1}",
                    name,
                    ms,
                    frags,
                    calls,
                    if *frags > 0 {
                        ms * 1.0e6 / *frags as f64
                    } else {
                        0.0
                    }
                );
            }
        }

        results
    }

    fn headless_draw_pass(
        &mut self,
        draw_pass_id: DrawPassId,
        render_threads: usize,
        parallel_min_tris: usize,
        fb: &mut Framebuffer,
        texture_cache: &mut TextureConversionCache,
        converted_textures: &mut Vec<crate::texture::TextureId>,
        mut profile: Option<&mut RenderProfile>,
    ) {
        let draw_list_id = match self.passes[draw_pass_id].main_draw_list_id {
            Some(id) => id,
            None => return,
        };

        let zbias_step = self.passes[draw_pass_id].zbias_step;
        let mut zbias = 0.0f32;

        // ── Modo BANDAS (`MAKEPAD_HEADLESS_BANDS=N`) ──────────────────────────
        //
        // POR QUÉ EXISTE: el pool de hilos que trae el backend headless también
        // parte por filas, pero lo hace DENTRO de cada draw-call. Con ~51
        // draw-calls por frame eso son 51 repartos y 51 barreras por frame, para
        // trocear triángulos que muchas veces cubren unos pocos cientos de
        // píxeles: el reparto cuesta más que el trabajo, y por eso "más hilos"
        // salía más lento.
        //
        // Este modo hace el reparto correcto para un rasterizador: se recorre la
        // escena UNA vez en serie (resolver shaders, uniforms, sombreado de
        // vértices) acumulando un plan de trabajo, y después N hilos rasterizan
        // el plan ENTERO, cada uno sobre su franja horizontal del framebuffer.
        // Una sola barrera por frame y cero contención: las franjas son
        // regiones disjuntas de memoria.
        let bands = std::env::var("MAKEPAD_HEADLESS_BANDS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 1);

        let mut jobs: Option<Vec<BandJob>> = bands.map(|_| Vec::new());

        self.headless_render_view(
            draw_pass_id,
            draw_list_id,
            &mut zbias,
            zbias_step,
            render_threads,
            parallel_min_tris,
            fb,
            texture_cache,
            converted_textures,
            profile.as_deref_mut(),
            &mut jobs,
        );

        if let (Some(n_bands), Some(jobs)) = (bands, jobs) {
            let raster_start = std::time::Instant::now();
            run_band_jobs(fb, &jobs, n_bands);
            if let Some(p) = profile.as_deref_mut() {
                p.raster_ms += raster_start.elapsed().as_secs_f64() * 1000.0;
            }
        }
    }

    fn headless_render_view(
        &mut self,
        draw_pass_id: DrawPassId,
        draw_list_id: DrawListId,
        zbias: &mut f32,
        zbias_step: f32,
        render_threads: usize,
        parallel_min_tris: usize,
        fb: &mut Framebuffer,
        texture_cache: &mut TextureConversionCache,
        converted_textures: &mut Vec<crate::texture::TextureId>,
        mut profile: Option<&mut RenderProfile>,
        jobs: &mut Option<Vec<BandJob>>,
    ) {
        let only_shader = std::env::var("MAKEPAD_HEADLESS_ONLY_SHADER").ok();
        let debug_text = std::env::var("MAKEPAD_HEADLESS_DEBUG_TEXT").is_ok();
        let draw_order_len = self.draw_lists[draw_list_id].draw_item_order_len();

        for order_index in 0..draw_order_len {
            let Some(draw_item_id) =
                self.draw_lists[draw_list_id].draw_item_id_at_order_index(order_index)
            else {
                continue;
            };
            let kind_tag = match &self.draw_lists[draw_list_id].draw_items[draw_item_id].kind {
                CxDrawKind::SubList(sub_id) => Some(*sub_id),
                CxDrawKind::DrawCall(_) => None,
                CxDrawKind::Empty => continue,
            };

            if let Some(sub_list_id) = kind_tag {
                let child_resets_zbias = self.draw_lists[sub_list_id].reset_zbias;
                let mut child_zbias = 0.0f32;
                self.headless_render_view(
                    draw_pass_id,
                    sub_list_id,
                    if child_resets_zbias {
                        &mut child_zbias
                    } else {
                        zbias
                    },
                    zbias_step,
                    render_threads,
                    parallel_min_tris,
                    fb,
                    texture_cache,
                    converted_textures,
                    profile.as_deref_mut(),
                    jobs,
                );
                continue;
            }

            let current_zbias = *zbias;
            {
                if let CxDrawKind::DrawCall(dc) =
                    &mut self.draw_lists[draw_list_id].draw_items[draw_item_id].kind
                {
                    dc.draw_call_uniforms.set_zbias(current_zbias);
                }
            }
            *zbias += zbias_step;

            let draw_item = &self.draw_lists[draw_list_id].draw_items[draw_item_id];
            let draw_call = match &draw_item.kind {
                CxDrawKind::DrawCall(dc) => dc,
                _ => continue,
            };

            let shader_id = draw_call.draw_shader_id;
            let sh = &self.draw_shaders.shaders[shader_id.index];
            let os_shader_id = match sh.os_shader_id {
                Some(id) => id,
                None => continue,
            };
            let is_draw_text_shader = match &sh.mapping.code {
                CxDrawShaderCode::Combined { code } => code.contains("sample_text_pixel"),
                CxDrawShaderCode::Separate { fragment, .. } => {
                    fragment.contains("sample_text_pixel")
                }
            };
            if let Some(only) = &only_shader {
                let keep = match only.as_str() {
                    "draw_text" => is_draw_text_shader,
                    _ => true,
                };
                if !keep {
                    continue;
                }
            }
            // ── Instrumentación H0-bis: cronómetro de PREPARACIÓN del draw-call ──
            // Cubre desde aquí hasta el shading de vértices: resolución de
            // símbolos, montaje de uniforms y relleno del `RenderCx`. Es coste
            // por draw-call, independiente del área, y es justo lo que NO se
            // ahorra con repintado parcial.
            let setup_start = std::time::Instant::now();
            let tex_ms_before = profile.as_deref().map(|p| p.texture_ms).unwrap_or(0.0);
            // Etiqueta legible del shader para el desglose. `debug_id` sale como
            // `0` cuando el shader viene del DSL (no tiene id de depuración), así
            // que se usa el id del shader compilado + si es el de texto, que es
            // la distinción que interesa (glifos vs Sdf2d).
            let shader_name = format!(
                "{}#{}",
                if is_draw_text_shader { "text" } else { "sdf" },
                os_shader_id
            );
            let os_shader = &self.draw_shaders.os_shaders[os_shader_id];
            let module = match &os_shader.module {
                Some(m) => m,
                None => continue,
            };

            // Load function pointers
            let vertex_fn: VertexFn = match module.symbol("makepad_headless_vertex") {
                Ok(f) => f,
                Err(_) => continue,
            };
            let fragment_fn: FragmentFn = match module.symbol("makepad_headless_fragment") {
                Ok(f) => f,
                Err(_) => continue,
            };

            // RenderCx layout info
            let rcx_size = os_shader.rcx_size;
            let rcx_vary_offset = os_shader.rcx_vary_offset;
            let rcx_quad_mode_offset = os_shader.rcx_quad_mode_offset;
            let rcx_frag_offset = os_shader.rcx_frag_offset;

            if rcx_size == 0 {
                continue;
            }

            // Per-draw-call RenderCx template (uniforms + textures) copied per worker.
            let rcx_f32s = rcx_size / std::mem::size_of::<f32>();
            let mut rcx_template = vec![0u8; rcx_size];

            // ── Per-draw-call: build uniform buffer arrays ──
            let draw_call_uniforms_slice = draw_call.draw_call_uniforms.as_slice();
            let pass_uniforms_slice = self.passes[draw_pass_id].pass_uniforms.as_slice();
            let draw_list_uniforms_slice =
                self.draw_lists[draw_list_id].draw_list_uniforms.as_slice();
            let dyn_uniforms = &draw_call.dyn_uniforms;
            let scope_buf = &sh.mapping.scope_uniforms_buf;
            let bindings = &sh.mapping.uniform_buffer_bindings;

            let max_buf_idx = bindings
                .bindings
                .iter()
                .map(|(_, idx)| *idx)
                .max()
                .unwrap_or(0);
            let dyn_buf_idx = max_buf_idx + 1;
            let scope_buf_idx = dyn_buf_idx + 1;
            let has_scope = !scope_buf.is_empty();
            let total_buffers = if has_scope {
                scope_buf_idx + 1
            } else {
                dyn_buf_idx + 1
            };

            const MAX_UNIFORM_BUFS: usize = 16;
            let total_buffers = total_buffers.min(MAX_UNIFORM_BUFS);
            let mut ptrs = [std::ptr::null::<f32>(); MAX_UNIFORM_BUFS];
            let mut lens = [0u32; MAX_UNIFORM_BUFS];

            for (type_name, idx) in &bindings.bindings {
                if *idx >= MAX_UNIFORM_BUFS {
                    continue;
                }
                if *type_name == id!(DrawCallUniforms) {
                    ptrs[*idx] = draw_call_uniforms_slice.as_ptr();
                    lens[*idx] = draw_call_uniforms_slice.len() as u32;
                } else if *type_name == id!(DrawPassUniforms) {
                    ptrs[*idx] = pass_uniforms_slice.as_ptr();
                    lens[*idx] = pass_uniforms_slice.len() as u32;
                } else if *type_name == id!(DrawListUniforms) {
                    ptrs[*idx] = draw_list_uniforms_slice.as_ptr();
                    lens[*idx] = draw_list_uniforms_slice.len() as u32;
                }
            }

            if dyn_buf_idx < MAX_UNIFORM_BUFS {
                ptrs[dyn_buf_idx] = dyn_uniforms.as_ptr();
                lens[dyn_buf_idx] = dyn_uniforms.len() as u32;
            }

            if has_scope && scope_buf_idx < MAX_UNIFORM_BUFS {
                ptrs[scope_buf_idx] = scope_buf.as_ptr();
                lens[scope_buf_idx] = scope_buf.len() as u32;
            }

            let uniform_count = total_buffers as u32;
            let uniform_ptrs = ptrs.as_ptr();
            let uniform_lens = lens.as_ptr();

            // ── Gather texture pointers, converting/caching to RGBA f32 when needed ──
            let mut tex_infos: Vec<[usize; 4]> = Vec::with_capacity(sh.mapping.textures.len());

            for tex_idx in 0..sh.mapping.textures.len() {
                if let Some(texture) = &draw_call.texture_slots[tex_idx] {
                    let texture_id = texture.texture_id();
                    let cxtexture = &self.textures[texture_id];
                    let __tex_t0 = std::time::Instant::now();
                    let __ya = converted_textures.contains(&texture_id);
                    let __info =
                        headless_texture_info(texture_id.0, cxtexture, texture_cache, __ya);
                    if !__ya {
                        converted_textures.push(texture_id);
                    }
                    if let Some(p) = profile.as_deref_mut() {
                        p.texture_ms += __tex_t0.elapsed().as_secs_f64() * 1000.0;
                    }
                    if let Some(info) = __info
                    {
                        tex_infos.push(info);
                    } else {
                        tex_infos.push([0, 0, 0, 0]);
                    }
                } else {
                    tex_infos.push([0, 0, 0, 0]);
                }
            }

            // ── Fill RenderCx buffer: uniforms + textures (per-draw-call, cold path) ──
            type FillUniformsFn = unsafe extern "C" fn(
                rcx_ptr: *mut f32,
                rcx_f32s: u32,
                uniform_ptrs: *const *const f32,
                uniform_lens: *const u32,
                uniform_count: u32,
                tex_infos_ptr: *const [usize; 4],
                tex_count: u32,
            );
            if let Ok(fill_fn) = module.symbol::<FillUniformsFn>("makepad_headless_fill_rcx") {
                unsafe {
                    fill_fn(
                        rcx_template.as_mut_ptr() as *mut f32,
                        rcx_f32s as u32,
                        uniform_ptrs,
                        uniform_lens,
                        uniform_count,
                        tex_infos.as_ptr(),
                        tex_infos.len() as u32,
                    );
                }
            }

            // Get geometry
            let geometry_id = match draw_call.geometry_id {
                Some(id) => id,
                None => continue,
            };
            let geom = &self.geometries[geometry_id];
            let vertices = &geom.vertices;
            let indices = &geom.indices;

            if indices.is_empty() || vertices.is_empty() {
                continue;
            }

            let instances_data = match &draw_item.instances {
                Some(data) => data.as_slice(),
                None => continue,
            };

            let total_instance_slots = sh.mapping.instances.total_slots;
            if total_instance_slots == 0 {
                continue;
            }
            let instance_count = instances_data.len() / total_instance_slots;
            if instance_count == 0 {
                continue;
            }
            if sh.mapping.flags.debug_draw {
                CxDrawShaderMapping::debug_dump_shader_draw_call(
                    "headless",
                    draw_item_id,
                    sh,
                    draw_call,
                    instances_data,
                    instance_count,
                );
            }

            let geom_slots = sh.mapping.geometries.total_slots;
            let varying_slots = sh.mapping.varying_total_slots;

            let vertex_count = if geom_slots > 0 {
                vertices.len() / geom_slots
            } else {
                0
            };
            if vertex_count == 0 {
                continue;
            }
            let tri_count = indices.len() / 3;
            if tri_count == 0 {
                continue;
            }
            if let Some(p) = profile.as_deref_mut() {
                p.draw_calls += 1;
                p.total_instances += instance_count;
                p.total_triangles += tri_count * instance_count;
            }

            if let Some(p) = profile.as_deref_mut() {
                // La conversión de texturas ya se contabiliza aparte: se resta
                // para que `setup_ms` sea preparación pura.
                let tex_delta = p.texture_ms - tex_ms_before;
                p.setup_ms += setup_start.elapsed().as_secs_f64() * 1000.0 - tex_delta;
            }

            let vertex_start = std::time::Instant::now();
            let shaded_vert_count = instance_count * vertex_count;
            let mut shaded_positions = vec![[0.0f32; 4]; shaded_vert_count];
            let mut shaded_varyings = vec![0.0f32; shaded_vert_count * varying_slots];

            for inst_idx in 0..instance_count {
                let inst_offset = inst_idx * total_instance_slots;
                let inst_slice = &instances_data[inst_offset..inst_offset + total_instance_slots];
                let inst_base = inst_idx * vertex_count;

                for vert_idx in 0..vertex_count {
                    let geom_offset = vert_idx * geom_slots;
                    let geom_slice = &vertices[geom_offset..geom_offset + geom_slots];
                    let shaded_idx = inst_base + vert_idx;
                    let vary_offset = shaded_idx * varying_slots;
                    let varying_out = &mut shaded_varyings
                        [vary_offset..vary_offset.saturating_add(varying_slots)];

                    unsafe {
                        vertex_fn(
                            geom_slice.as_ptr(),
                            geom_slice.len() as u32,
                            inst_slice.as_ptr(),
                            inst_slice.len() as u32,
                            uniform_ptrs,
                            uniform_lens,
                            uniform_count,
                            varying_out.as_mut_ptr(),
                            varying_slots as u32,
                            &mut shaded_positions[shaded_idx],
                        );
                    }
                }
            }
            if let Some(p) = profile.as_deref_mut() {
                p.vertex_ms += vertex_start.elapsed().as_secs_f64() * 1000.0;
            }

            let flat_slots = os_shader.flat_varying_slots.min(varying_slots);
            // `MAKEPAD_HEADLESS_NO_DERIV` (instrumentación H0-bis): desactiva el
            // camino de derivadas. NO es una optimización usable —el texto pierde
            // el antialiasing—, es una sonda: ese camino invoca el shader de
            // fragmento TRES veces por píxel (dFdx, dFdy y el real) e interpola
            // los varyings tres veces. Comparar con/sin dice cuánto del coste del
            // texto es el shader en sí y cuánto es el andamiaje de derivadas.
            let uses_derivatives =
                os_shader.uses_derivatives && std::env::var("MAKEPAD_HEADLESS_NO_DERIV").is_err();
            let row_chunks = compute_row_chunks(fb.height, render_threads);
            let use_parallel = row_chunks.len() > 1
                && tri_count.saturating_mul(instance_count) >= parallel_min_tris
                && self.os.render_pool.is_some();
            if let Some(p) = profile.as_deref_mut() {
                if use_parallel {
                    p.parallel_draw_calls += 1;
                } else {
                    p.serial_draw_calls += 1;
                }
            }

            // Modo bandas: no se rasteriza aquí. Se guarda el draw-call ya
            // preparado y se pinta al final del pase, con N hilos por franja.
            if let Some(jobs) = jobs.as_mut() {
                jobs.push(BandJob {
                    indices: indices.clone(),
                    instance_count,
                    vertex_count,
                    varying_slots,
                    shaded_positions,
                    shaded_varyings,
                    flat_slots,
                    rcx_template,
                    rcx_size,
                    rcx_f32s,
                    rcx_vary_offset,
                    rcx_quad_mode_offset,
                    rcx_frag_offset,
                    uses_derivatives,
                    fragment_fn,
                    is_draw_text_shader,
                });
                continue;
            }

            let raster_start = std::time::Instant::now();
            let shaded_before =
                super::virtual_gpu::FRAG_SHADED.load(std::sync::atomic::Ordering::Relaxed);
            if use_parallel {
                let pool = self.os.render_pool.as_ref().unwrap();
                let (done_tx, done_rx) = mpsc::channel::<()>();
                let width = fb.width;
                let height = fb.height;
                let color_ptr = fb.color.as_mut_ptr() as usize;
                let depth_ptr = fb.depth.as_mut_ptr() as usize;
                let indices_ptr = indices.as_ptr() as usize;
                let indices_len = indices.len();
                let shaded_positions_ptr = shaded_positions.as_ptr() as usize;
                let shaded_positions_len = shaded_positions.len();
                let shaded_varyings_ptr = shaded_varyings.as_ptr() as usize;
                let shaded_varyings_len = shaded_varyings.len();
                let rcx_template_ptr = rcx_template.as_ptr() as usize;
                let rcx_template_len = rcx_template.len();

                for chunk in row_chunks.iter().copied() {
                    let done_tx = done_tx.clone();
                    pool.execute(move |_| {
                        let row_start = chunk.start;
                        let row_end = chunk.end;
                        let row_count = row_end.saturating_sub(row_start);
                        if row_count == 0 {
                            let _ = done_tx.send(());
                            return;
                        }

                        let pixel_offset = row_start * width;
                        let pixel_count = row_count * width;
                        let color_chunk = unsafe {
                            std::slice::from_raw_parts_mut(
                                (color_ptr as *mut [f32; 4]).add(pixel_offset),
                                pixel_count,
                            )
                        };
                        let depth_chunk = unsafe {
                            std::slice::from_raw_parts_mut(
                                (depth_ptr as *mut f32).add(pixel_offset),
                                pixel_count,
                            )
                        };
                        let indices = unsafe {
                            std::slice::from_raw_parts(indices_ptr as *const u32, indices_len)
                        };
                        let shaded_positions = unsafe {
                            std::slice::from_raw_parts(
                                shaded_positions_ptr as *const [f32; 4],
                                shaded_positions_len,
                            )
                        };
                        let shaded_varyings = unsafe {
                            std::slice::from_raw_parts(
                                shaded_varyings_ptr as *const f32,
                                shaded_varyings_len,
                            )
                        };
                        let rcx_template = unsafe {
                            std::slice::from_raw_parts(
                                rcx_template_ptr as *const u8,
                                rcx_template_len,
                            )
                        };

                        rasterize_instances_rows(
                            color_chunk,
                            depth_chunk,
                            width,
                            height,
                            row_start,
                            row_end,
                            indices,
                            instance_count,
                            vertex_count,
                            varying_slots,
                            shaded_positions,
                            shaded_varyings,
                            flat_slots,
                            rcx_template,
                            rcx_size,
                            rcx_f32s,
                            rcx_vary_offset,
                            rcx_quad_mode_offset,
                            rcx_frag_offset,
                            uses_derivatives,
                            fragment_fn,
                            debug_text,
                            is_draw_text_shader,
                        );

                        let _ = done_tx.send(());
                    });
                }

                drop(done_tx);
                for _ in 0..row_chunks.len() {
                    if done_rx.recv().is_err() {
                        break;
                    }
                }
            } else {
                rasterize_instances_rows(
                    fb.color.as_mut_slice(),
                    fb.depth.as_mut_slice(),
                    fb.width,
                    fb.height,
                    0,
                    fb.height,
                    indices,
                    instance_count,
                    vertex_count,
                    varying_slots,
                    &shaded_positions,
                    &shaded_varyings,
                    flat_slots,
                    &rcx_template,
                    rcx_size,
                    rcx_f32s,
                    rcx_vary_offset,
                    rcx_quad_mode_offset,
                    rcx_frag_offset,
                    uses_derivatives,
                    fragment_fn,
                    debug_text,
                    is_draw_text_shader,
                );
            }
            if let Some(p) = profile.as_deref_mut() {
                let dt = raster_start.elapsed().as_secs_f64() * 1000.0;
                let shaded = super::virtual_gpu::FRAG_SHADED
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .saturating_sub(shaded_before);
                p.raster_ms += dt;
                let e = p.per_shader.entry(shader_name).or_insert((0.0, 0, 0));
                e.0 += dt;
                e.1 += shaded;
                e.2 += 1;
            }
        }
    }
}

/// Copy varying data into the rcx buffer at the given offset.
#[inline]
fn write_varyings(
    rcx_buf: &mut [u8],
    offset: usize,
    varyings: &[f32],
    vary_bytes: usize,
    rcx_size: usize,
) {
    if offset + vary_bytes <= rcx_size {
        unsafe {
            std::ptr::copy_nonoverlapping(
                varyings.as_ptr() as *const u8,
                rcx_buf.as_mut_ptr().add(offset),
                vary_bytes,
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PNG encoding
// ─────────────────────────────────────────────────────────────────────────────

pub fn encode_png_rgba(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| "rgba size overflow while encoding png".to_string())?;
    if rgba.len() != expected {
        return Err(format!(
            "encode_png_rgba: expected {} bytes, got {}",
            expected,
            rgba.len()
        ));
    }

    let options = EncoderOptions::default()
        .set_width(width as usize)
        .set_height(height as usize)
        .set_depth(BitDepth::Eight)
        .set_colorspace(ColorSpace::RGBA);

    let mut encoder = PngEncoder::new(rgba, options);
    let mut out = Vec::new();
    encoder
        .encode(&mut out)
        .map_err(|err| format!("headless png encode failed: {err:?}"))?;
    Ok(out)
}
