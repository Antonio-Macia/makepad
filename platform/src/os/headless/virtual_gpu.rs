/// A software rasterizer that interpolates float varyings and calls a fragment
/// shader callback per pixel.

// ─────────────────────────────────────────────────────────────────────────────
// Instrumentación H0-bis (ATLAS): scissor global + contadores de fragmentos
// ─────────────────────────────────────────────────────────────────────────────
//
// POR QUÉ: para decidir si un escritorio por CPU es viable hay que saber cuánto
// cuesta repintar SÓLO una ventana en vez de la pantalla entera. Makepad no
// tiene hoy repintado parcial (ni damage ni scissor) en el backend headless, así
// que se añade aquí el mecanismo mínimo para MEDIRLO: un rectángulo global que
// recorta la caja envolvente de cada triángulo. No es damage tracking de verdad
// (no evita recorrer la escena), pero aísla exactamente el coste por píxel, que
// es la variable que decide.
//
// `MAKEPAD_HEADLESS_CLIP=x,y,w,h` en píxeles de dispositivo.

/// Suspende el recorte por daño mientras no haya un frame anterior que conservar.
///
/// # Por qué hace falta
///
/// El daño dice «esto es lo único que ha cambiado», y esa frase **presupone que
/// lo demás sigue en pantalla**. En el primer frame, y tras cada redimensión, no
/// sigue: el framebuffer acaba de nacer. Si el recorte se aplicara igualmente,
/// todo lo de fuera del rectángulo no se pintaría nunca — ni en ese frame ni en
/// ninguno, porque los siguientes tampoco lo tocan.
///
/// Se destapó midiendo (2026-08-16): con el framebuffer persistente ya puesto,
/// presentar la pantalla entera en vez de sólo el daño daba **904.000 píxeles de
/// diferencia** sobre 1.024.000. O sea que el framebuffer «persistente» estaba
/// conservando, fielmente, un vacío.
static CLIP_SUSPENDED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Enciende o apaga la suspensión del recorte. La llama el renderizador al
/// decidir si el framebuffer de la ventana se reutiliza o es nuevo.
pub fn set_clip_suspended(suspended: bool) {
    CLIP_SUSPENDED.store(suspended, std::sync::atomic::Ordering::Relaxed);
}

/// Rectángulo de recorte global `(x0, y0, x1, y1)` inclusivo-exclusivo, o `None`.
///
/// Devuelve `None` —o sea, «pinta entero»— mientras el recorte esté suspendido
/// (ver [`set_clip_suspended`]), aunque la variable de entorno esté puesta.
pub fn headless_clip_rect() -> Option<(i32, i32, i32, i32)> {
    if CLIP_SUSPENDED.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    static CLIP: std::sync::OnceLock<Option<(i32, i32, i32, i32)>> = std::sync::OnceLock::new();
    *CLIP.get_or_init(|| {
        let raw = std::env::var("MAKEPAD_HEADLESS_CLIP").ok()?;
        let parts: Vec<i32> = raw.split(',').filter_map(|p| p.trim().parse().ok()).collect();
        if parts.len() != 4 {
            return None;
        }
        Some((parts[0], parts[1], parts[0] + parts[2], parts[1] + parts[3]))
    })
}

/// Píxeles visitados dentro de la caja envolvente (candidatos a fragmento).
pub static FRAG_TESTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Invocaciones REALES del shader de fragmento (pasaron cobertura y z-test).
pub static FRAG_SHADED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub struct Framebuffer {
    pub width: usize,
    pub height: usize,
    pub color: Vec<[f32; 4]>, // RGBA linear premultiplied
    pub depth: Vec<f32>,
}

impl Framebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let pixels = width * height;
        Self {
            width,
            height,
            color: vec![[0.0; 4]; pixels],
            depth: vec![1.0; pixels],
        }
    }

    pub fn clear(&mut self, color: [f32; 4], depth: f32) {
        self.color.fill(color);
        self.depth.fill(depth);
    }

    /// Limpia sólo un rectángulo del framebuffer (repintado parcial simulado).
    pub fn clear_rect(
        &mut self,
        color: [f32; 4],
        depth: f32,
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
    ) {
        let x0 = x0.max(0) as usize;
        let y0 = y0.max(0) as usize;
        let x1 = (x1.max(0) as usize).min(self.width);
        let y1 = (y1.max(0) as usize).min(self.height);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        for y in y0..y1 {
            let base = y * self.width;
            self.color[base + x0..base + x1].fill(color);
            self.depth[base + x0..base + x1].fill(depth);
        }
    }

    /// Convierte a RGBA8 **sólo el rectángulo de recorte**, reutilizando `out`.
    ///
    /// 🔴 POR QUÉ EXISTE. `to_rgba8()` convierte la pantalla ENTERA y reserva un
    /// `Vec` nuevo en cada frame. Medido en ATLAS/H0 a 1280×720: **9-12 ms por
    /// frame**, sin importar que el daño sea de 100×20 píxeles. Con un ciclo
    /// completo de 16,5 ms, eso era **más de la mitad del presupuesto de 60 Hz
    /// gastado en volver a convertir píxeles que no habían cambiado** — la misma
    /// clase de defecto que la caché de texturas que nunca acertaba: trabajo de
    /// pantalla completa dentro de un repintado parcial.
    ///
    /// `out` se conserva entre frames a propósito: lo de fuera del recorte es
    /// justamente lo que NO hay que volver a tocar.
    ///
    /// ⚠ Ojo a lo que esto NO arregla: el `Framebuffer` sí se recrea y se limpia
    /// en cada frame, así que el recorte sigue siendo un instrumento de medida y
    /// no seguimiento de daño de verdad. Para eso hace falta además que el
    /// framebuffer persista.
    pub fn to_rgba8_into(&self, out: &mut Vec<u8>, clip: Option<(i32, i32, i32, i32)>) {
        let needed = self.width * self.height * 4;
        if out.len() != needed {
            out.clear();
            out.resize(needed, 0);
        }
        let (x0, y0, x1, y1) = match clip {
            Some((x0, y0, x1, y1)) => (
                x0.max(0) as usize,
                y0.max(0) as usize,
                (x1.max(0) as usize).min(self.width),
                (y1.max(0) as usize).min(self.height),
            ),
            None => (0, 0, self.width, self.height),
        };
        for y in y0..y1 {
            let fila = y * self.width;
            for x in x0..x1 {
                let c = &self.color[fila + x];
                // premultiplicado → sin premultiplicar, para la salida
                let a = c[3].clamp(0.0, 1.0);
                let inv_a = if a > 0.0 { 1.0 / a } else { 0.0 };
                let base = (fila + x) * 4;
                out[base] = ((c[0] * inv_a).clamp(0.0, 1.0) * 255.0).round() as u8;
                out[base + 1] = ((c[1] * inv_a).clamp(0.0, 1.0) * 255.0).round() as u8;
                out[base + 2] = ((c[2] * inv_a).clamp(0.0, 1.0) * 255.0).round() as u8;
                out[base + 3] = (a * 255.0).round() as u8;
            }
        }
    }

    pub fn to_rgba8(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.width * self.height * 4];
        for (i, c) in self.color.iter().enumerate() {
            // c is premultiplied alpha - unpremultiply for PNG
            let a = c[3].clamp(0.0, 1.0);
            let inv_a = if a > 0.0 { 1.0 / a } else { 0.0 };
            let r = (c[0] * inv_a).clamp(0.0, 1.0);
            let g = (c[1] * inv_a).clamp(0.0, 1.0);
            let b = (c[2] * inv_a).clamp(0.0, 1.0);
            let base = i * 4;
            out[base] = (r * 255.0).round() as u8;
            out[base + 1] = (g * 255.0).round() as u8;
            out[base + 2] = (b * 255.0).round() as u8;
            out[base + 3] = (a * 255.0).round() as u8;
        }
        out
    }
}

/// Per-fragment derivative deltas.
/// `dvary_dx[i]` ~= varying(i) at (x+1,y) minus current varying(i),
/// `dvary_dy[i]` ~= varying(i) at (x,y+1) minus current varying(i).
#[derive(Default)]
pub struct TriangleDerivatives {
    pub dvary_dx: Vec<f32>,
    pub dvary_dy: Vec<f32>,
}

#[derive(Default)]
pub struct RasterScratch {
    pub interp: Vec<f32>,
    pub interp_dx: Vec<f32>,
    pub interp_dy: Vec<f32>,
    pub derivs: TriangleDerivatives,
}

impl RasterScratch {
    fn ensure_vary_len(&mut self, vary_len: usize, compute_derivatives: bool) {
        if self.interp.len() < vary_len {
            self.interp.resize(vary_len, 0.0);
        }
        if compute_derivatives {
            if self.interp_dx.len() < vary_len {
                self.interp_dx.resize(vary_len, 0.0);
            }
            if self.interp_dy.len() < vary_len {
                self.interp_dy.resize(vary_len, 0.0);
            }
            if self.derivs.dvary_dx.len() < vary_len {
                self.derivs.dvary_dx.resize(vary_len, 0.0);
            }
            if self.derivs.dvary_dy.len() < vary_len {
                self.derivs.dvary_dy.resize(vary_len, 0.0);
            }
        }
    }
}

/// Rasterize only a row range `[row_start, row_end)` of the framebuffer.
/// `color`/`depth_buf` are row-contiguous slices sized `(row_end-row_start)*width`.
pub fn rasterize_triangle_rows<F>(
    width: usize,
    height: usize,
    row_start: usize,
    row_end: usize,
    color: &mut [[f32; 4]],
    depth_buf: &mut [f32],
    p0: &[f32; 4],
    vary0: &[f32],
    p1: &[f32; 4],
    vary1: &[f32],
    p2: &[f32; 4],
    vary2: &[f32],
    flat_slots: usize,
    compute_derivatives: bool,
    scratch: &mut RasterScratch,
    fragment_fn: &mut F,
) where
    F: FnMut(&[f32], &TriangleDerivatives, u32, u32, i32, i32) -> Option<[f32; 4]>,
{
    if width == 0 || height == 0 {
        return;
    }
    let row_start = row_start.min(height);
    let row_end = row_end.min(height);
    if row_start >= row_end {
        return;
    }
    let expected_len = (row_end - row_start) * width;
    if color.len() < expected_len || depth_buf.len() < expected_len {
        return;
    }
    if vary0.len() != vary1.len() || vary1.len() != vary2.len() {
        return;
    }

    let w = width as f32;
    let h = height as f32;

    // Convert from clip space [-1,1] to screen space [0, width/height].
    let ndc_to_screen = |pos: &[f32; 4]| -> (f32, f32, f32) {
        let inv_w = if pos[3] != 0.0 { 1.0 / pos[3] } else { 1.0 };
        let ndc_x = pos[0] * inv_w;
        let ndc_y = pos[1] * inv_w;
        let ndc_z = pos[2] * inv_w;
        let sx = (ndc_x * 0.5 + 0.5) * w;
        let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * h; // flip Y
                                                  // Makepad shaders output depth in [0, 1] clip space in practice.
                                                  // Keep it as-is to avoid collapsing depth precision.
        let sz = ndc_z;
        (sx, sy, sz)
    };

    let (sx0, sy0, sz0) = ndc_to_screen(p0);
    let (sx1, sy1, sz1) = ndc_to_screen(p1);
    let (sx2, sy2, sz2) = ndc_to_screen(p2);

    let mut sx = [sx0, sx1, sx2];
    let mut sy = [sy0, sy1, sy2];
    let mut sz = [sz0, sz1, sz2];
    let mut inv_clip_w = [
        if p0[3].abs() > f32::EPSILON {
            1.0 / p0[3]
        } else {
            0.0
        },
        if p1[3].abs() > f32::EPSILON {
            1.0 / p1[3]
        } else {
            0.0
        },
        if p2[3].abs() > f32::EPSILON {
            1.0 / p2[3]
        } else {
            0.0
        },
    ];
    let mut vary_src = [vary0, vary1, vary2];

    // Ensure a positive area so a single top-left rule works for all triangles.
    let mut area = edge(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2]);
    if area.abs() <= f32::EPSILON {
        return;
    }
    if area < 0.0 {
        sx.swap(1, 2);
        sy.swap(1, 2);
        sz.swap(1, 2);
        inv_clip_w.swap(1, 2);
        vary_src.swap(1, 2);
        area = -area;
    }

    let mut min_x = sx[0].min(sx[1]).min(sx[2]).floor().max(0.0) as i32;
    let mut min_y = sy[0].min(sy[1]).min(sy[2]).floor().max(row_start as f32) as i32;
    let mut max_x = sx[0].max(sx[1]).max(sx[2]).ceil().min(w - 1.0) as i32;
    let mut max_y = sy[0].max(sy[1]).max(sy[2]).ceil().min(row_end as f32 - 1.0) as i32;

    // Scissor global (instrumentación H0-bis): recorta la caja envolvente al
    // rectángulo de damage simulado. Todo lo de fuera ni siquiera se visita.
    if let Some((cx0, cy0, cx1, cy1)) = headless_clip_rect() {
        min_x = min_x.max(cx0);
        min_y = min_y.max(cy0);
        max_x = max_x.min(cx1 - 1);
        max_y = max_y.min(cy1 - 1);
    }

    if max_x < min_x || max_y < min_y {
        return;
    }
    FRAG_TESTED.fetch_add(
        ((max_x - min_x + 1) as u64) * ((max_y - min_y + 1) as u64),
        std::sync::atomic::Ordering::Relaxed,
    );

    let vary_len = vary_src[0].len();
    let flat_slots = flat_slots.min(vary_len);
    scratch.ensure_vary_len(vary_len, compute_derivatives);
    let empty_derivs = TriangleDerivatives::default();

    let inv_area = 1.0 / area;

    // Edge increments for stepping one pixel in +x/+y.
    let e0_dx = sy[2] - sy[1];
    let e1_dx = sy[0] - sy[2];
    let e2_dx = sy[1] - sy[0];
    let e0_dy = sx[1] - sx[2];
    let e1_dy = sx[2] - sx[0];
    let e2_dy = sx[0] - sx[1];

    let top_left_0 = is_top_left(sx[1], sy[1], sx[2], sy[2]);
    let top_left_1 = is_top_left(sx[2], sy[2], sx[0], sy[0]);
    let top_left_2 = is_top_left(sx[0], sy[0], sx[1], sy[1]);

    let interpolate_perspective = |w0: f32, w1: f32, w2: f32, out: &mut [f32]| -> bool {
        let a0 = w0 * inv_clip_w[0];
        let a1 = w1 * inv_clip_w[1];
        let a2 = w2 * inv_clip_w[2];
        let denom = a0 + a1 + a2;
        if denom.abs() <= f32::EPSILON {
            return false;
        }
        let inv_denom = 1.0 / denom;
        for i in 0..vary_len {
            out[i] = (a0 * vary_src[0][i] + a1 * vary_src[1][i] + a2 * vary_src[2][i]) * inv_denom;
        }
        true
    };

    // Contador local (no atómico por píxel: el fetch_add por fragmento falsearía
    // la medida). Se vuelca una sola vez al terminar el triángulo.
    let mut shaded_here: u64 = 0;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            let e0 = edge(sx[1], sy[1], sx[2], sy[2], px, py);
            let e1 = edge(sx[2], sy[2], sx[0], sy[0], px, py);
            let e2 = edge(sx[0], sy[0], sx[1], sy[1], px, py);

            // GPU-like top-left rule avoids shared-edge gaps and overlaps.
            if !edge_pass(e0, top_left_0)
                || !edge_pass(e1, top_left_1)
                || !edge_pass(e2, top_left_2)
            {
                continue;
            }

            let w0 = e0 * inv_area;
            let w1 = e1 * inv_area;
            let w2 = e2 * inv_area;

            let depth = sz[0] * w0 + sz[1] * w1 + sz[2] * w2;
            let local_y = y as usize - row_start;
            let index = local_y * width + x as usize;

            // Depth test (less-or-equal for overlapping widgets with same zbias)
            if depth > depth_buf[index] {
                continue;
            }

            if !interpolate_perspective(w0, w1, w2, &mut scratch.interp[..vary_len]) {
                continue;
            }

            let lane_x = (x as u32) & 1;
            let lane_y = (y as u32) & 1;
            // Dyn/rust instance slots are constant across the primitive.
            // Keep them bit-stable (no interpolation drift) for shader equality tests.
            for i in 0..flat_slots {
                scratch.interp[i] = vary_src[0][i];
            }

            shaded_here += 1;
            let frag_color = if compute_derivatives {
                // Build dFdx/dFdy-style deltas by evaluating at neighboring pixel centers.
                // GPU derivatives are pairwise across a 2x2 quad:
                // dFdx for odd x lanes uses (current - left), even x uses (right - current).
                // dFdy for odd y lanes uses (current - up), even y uses (down - current).
                let dx_sign = if lane_x == 0 { 1.0 } else { -1.0 };
                let dy_sign = if lane_y == 0 { 1.0 } else { -1.0 };

                let wx0 = (e0 + dx_sign * e0_dx) * inv_area;
                let wx1 = (e1 + dx_sign * e1_dx) * inv_area;
                let wx2 = (e2 + dx_sign * e2_dx) * inv_area;
                let wy0 = (e0 + dy_sign * e0_dy) * inv_area;
                let wy1 = (e1 + dy_sign * e1_dy) * inv_area;
                let wy2 = (e2 + dy_sign * e2_dy) * inv_area;

                if !interpolate_perspective(wx0, wx1, wx2, &mut scratch.interp_dx[..vary_len])
                    || !interpolate_perspective(wy0, wy1, wy2, &mut scratch.interp_dy[..vary_len])
                {
                    continue;
                }

                for i in 0..vary_len {
                    scratch.derivs.dvary_dx[i] = scratch.interp_dx[i] - scratch.interp[i];
                    scratch.derivs.dvary_dy[i] = scratch.interp_dy[i] - scratch.interp[i];
                }
                for i in 0..flat_slots {
                    scratch.derivs.dvary_dx[i] = 0.0;
                    scratch.derivs.dvary_dy[i] = 0.0;
                }

                match fragment_fn(
                    &scratch.interp[..vary_len],
                    &scratch.derivs,
                    lane_x,
                    lane_y,
                    x,
                    y,
                ) {
                    Some(c) => c,
                    None => continue,
                }
            } else {
                match fragment_fn(
                    &scratch.interp[..vary_len],
                    &empty_derivs,
                    lane_x,
                    lane_y,
                    x,
                    y,
                ) {
                    Some(c) => c,
                    None => continue,
                }
            };

            // Premultiplied alpha blending (source-over)
            let src_a = frag_color[3];
            let dst = color[index];
            color[index] = blend_premul_src_over(frag_color, dst);
            // Match common UI blending behavior: fully transparent pixels should
            // not occlude subsequent geometry in depth.
            if src_a > 0.02 {
                depth_buf[index] = depth;
            }
        }
    }
    FRAG_SHADED.fetch_add(shaded_here, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (px - ax) * (by - ay) - (py - ay) * (bx - ax)
}

#[inline]
fn is_top_left(ax: f32, ay: f32, bx: f32, by: f32) -> bool {
    let dy = by - ay;
    let dx = bx - ax;
    // Screen-space Y grows downward, so top-left differs from Y-up convention.
    dy > 0.0 || (dy == 0.0 && dx < 0.0)
}

#[inline]
fn edge_pass(edge_value: f32, top_left: bool) -> bool {
    const EDGE_EPS: f32 = 1.0e-6;
    if edge_value < -EDGE_EPS {
        false
    } else if edge_value > 0.0 {
        true
    } else {
        top_left
    }
}

#[inline]
fn blend_premul_src_over(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let inv_src_a = 1.0 - src[3];
    [
        src[0] + dst[0] * inv_src_a,
        src[1] + dst[1] * inv_src_a,
        src[2] + dst[2] * inv_src_a,
        src[3] + dst[3] * inv_src_a,
    ]
}
