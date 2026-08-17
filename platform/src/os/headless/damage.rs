//! Cálculo del daño: qué rectángulo de la pantalla ha cambiado este frame.
//!
//! # Qué problema resuelve
//!
//! El backend por software rasteriza por píxel, así que repintar la pantalla
//! entera cuando sólo parpadea un cursor es la diferencia medida entre **28,4 ms**
//! y **8,0 ms** — o sea, entre llegar a la cadencia del monitor y no llegar.
//!
//! Hasta ahora el recorte por daño existía pero se **declaraba a mano**
//! (`MAKEPAD_HEADLESS_CLIP=x,y,w,h`): servía para MEDIR el ahorro, no para
//! obtenerlo. Este módulo lo calcula.
//!
//! # De dónde sale el dato
//!
//! makepad ya sabe qué hay que repintar, sólo que con granularidad de **draw
//! list**, no de rectángulo: `DrawEvent` trae `redraw_all`, `draw_lists` y
//! `draw_lists_and_children`. Y cada `CxDrawList` lleva ahora su `painted_rect`,
//! que es la caja que ocupa en pantalla.
//!
//! El daño es entonces la unión de esas cajas, para las listas que se van a
//! repintar.
//!
//! # 🔴 La fuente correcta, y la que parece serlo y no lo es
//!
//! El primer intento usó `CxDrawList::rect_areas`, y **estaba mal**. Esa lista es
//! la contabilidad de áreas ALINEADAS (texto, hit-testing): no contiene los fondos
//! ni la mayoría de lo que se pinta. El daño salía demasiado pequeño y el
//! resultado no fue un error, fueron píxeles sin refrescar — el oráculo lo cazó
//! con **13 de 14 frames distintos**.
//!
//! `painted_rect` sale de `dirty_check_rect`, el rectángulo del turtle que makepad
//! ya usaba para decidir si una lista se repinta. Estaba en el lado de dibujo y no
//! en el de plataforma, así que el rasterizador sabía QUÉ se repintaba pero no
//! DÓNDE.
//!
//! # Y la condición que hace falta en la UI
//!
//! El daño no puede ser más fino que las draw lists que existan. Un `View`
//! normal dibuja dentro de la lista de su padre, así que **una ventana entera
//! puede ser UNA sola draw list** — medido: `atlas-ui` lo era, y el daño salía
//! siempre «pantalla entera» por mucho que el cálculo fuera correcto. Lo que
//! deba repintarse aparte necesita `new_batch: true` en el DSL. Es una decisión
//! de la UI, no del backend, y por eso está escrito aquí: sin ella, esto no
//! ahorra nada y parece que no funciona.
//!
//! # 🔴 La trampa: también hay que borrar donde ESTABA
//!
//! Si un widget se encoge, se mueve o desaparece, su rectángulo NUEVO no cubre el
//! sitio donde estaba pintado. Con framebuffer persistente, lo de antes **sigue en
//! pantalla**: quedaría un fantasma. Por eso se guarda el rectángulo del frame
//! anterior de cada lista y el daño une **el de antes y el de ahora**.
//!
//! Esto no es teórico y es el fallo clásico del repintado parcial: se ve como
//! restos de una ventana que ya no está, y sólo aparece al mover cosas — nunca en
//! una captura estática, que es justo lo que suele mirarse.
//!
//! # ✅ Un defecto PRE-EXISTENTE que este oráculo destapó, y que ya está ARREGLADO
//!
//! El backend por software pintaba **la primera fila de texto más apagada en el
//! frame 0** que en todos los siguientes: 1.109 px, hasta 77 por canal.
//!
//! **La causa:** el rasterizado de glifos es asíncrono
//! (`draw/src/text/fonts.rs`, `dispatch_msdf_jobs` / `apply_completed_msdf_jobs`)
//! y el atlas se sube a la textura por rectángulo sucio. El primer frame se
//! pintaba con el atlas a medio asentar.
//!
//! **Por qué había que arreglarlo y no ignorarlo**, aunque con repintado completo
//! se autocorrigiera en el frame 1: toda **captura de UN solo frame** recogía el
//! texto sin asentar. O sea que cualquier screenshot de referencia nacía mal, y
//! una referencia mala no da error — da diferencias falsas en todo lo que se
//! compare con ella. Y con daño se congelaba, porque nadie vuelve a tocar esa
//! zona.
//!
//! **El arreglo:** un ciclo completo de dibujo+render que no se presenta, antes
//! del primer frame (`os/headless/event_loop.rs`, `warmup_done`). Verificado: la
//! diferencia entre el frame 0 y los siguientes en una escena estática pasa de
//! 1.109 px a **0**.
//!
//! Lo que quedó descartado por el camino, para no repetir el rato: no fue la
//! caché de conversión de texturas, ni el crecimiento del atlas por caracteres
//! nuevos, ni el framebuffer persistente, ni el propio daño.
//!
//! # Cómo se comprueba que no miente
//!
//! Con un oráculo que no sale de aquí: `MAKEPAD_HEADLESS_DAMAGE=0` desactiva el
//! cálculo y repinta entero. Si el daño es correcto, las dos corridas producen
//! imágenes **idénticas píxel a píxel**; cualquier diferencia es exactamente lo
//! que el daño se dejó fuera. Ver `damage_oracle` en `tools/`.

use crate::cx::Cx;
use crate::draw_list::DrawListId;
use crate::event::DrawEvent;
use std::collections::HashMap;

/// Rectángulo en píxeles de dispositivo, inclusivo-exclusivo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageRect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl DamageRect {
    /// Une dos rectángulos (la caja que contiene a los dos).
    ///
    /// Se une en vez de llevar una lista de rectángulos disjuntos a propósito: el
    /// scissor del rasterizador es UN rectángulo, así que una región compleja
    /// habría que convertirla igualmente a su caja envolvente. Cuando el
    /// rasterizador acepte varios rectángulos, esto crece; hoy sería complejidad
    /// sin destino.
    pub fn union(self, other: DamageRect) -> DamageRect {
        DamageRect {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    /// `true` si el rectángulo no encierra ningún píxel.
    pub fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }

    /// Recorta a los límites de la pantalla.
    pub fn clamp(self, width: i32, height: i32) -> DamageRect {
        DamageRect {
            x0: self.x0.max(0),
            y0: self.y0.max(0),
            x1: self.x1.min(width),
            y1: self.y1.min(height),
        }
    }

    /// Área en píxeles. Para los contadores: es la cifra que dice si el daño está
    /// ahorrando algo o es un adorno caro.
    pub fn area(&self) -> u64 {
        if self.is_empty() {
            0
        } else {
            (self.x1 - self.x0) as u64 * (self.y1 - self.y0) as u64
        }
    }
}

/// Estado del cálculo de daño entre frames.
#[derive(Default)]
pub struct DamageTracker {
    /// Rectángulo que ocupó cada draw list la última vez que se dibujó, en
    /// píxeles de dispositivo. Es la mitad que impide los fantasmas.
    anterior: HashMap<DrawListId, DamageRect>,
    /// Listas que este frame se van a repintar, tomadas del `DrawEvent` ANTES de
    /// despacharlo (`call_draw_event` lo consume con un `swap`).
    pendientes: Vec<DrawListId>,
    /// Con hijos incluidos.
    pendientes_con_hijos: Vec<DrawListId>,
    /// Repintado completo pedido explícitamente.
    todo: bool,
}

/// ¿Está activo el cálculo de daño?
///
/// `MAKEPAD_HEADLESS_DAMAGE=0` lo apaga — es el interruptor del oráculo, no una
/// opción de uso. Por defecto está **apagado** mientras no se haya verificado en
/// movimiento (ver la nota de la cabecera): un daño mal calculado no da error, deja
/// fantasmas, y eso es peor que no tener daño.
pub fn damage_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("MAKEPAD_HEADLESS_DAMAGE").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

impl DamageTracker {
    /// Anota qué se va a repintar. Se llama **antes** de `call_draw_event`, que es
    /// donde makepad vacía `new_draw_event` con un `mem::swap`; después ya no hay
    /// forma de saberlo.
    pub fn observar(&mut self, ev: &DrawEvent) {
        if std::env::var("MAKEPAD_HEADLESS_DAMAGE_TRACE").is_ok() {
            crate::log!(
                "[damage] observado: redraw_all={} listas={} listas_con_hijos={}",
                ev.redraw_all,
                ev.draw_lists.len(),
                ev.draw_lists_and_children.len()
            );
        }
        self.todo = ev.redraw_all;
        self.pendientes.clear();
        self.pendientes.extend_from_slice(&ev.draw_lists);
        self.pendientes_con_hijos.clear();
        self.pendientes_con_hijos
            .extend_from_slice(&ev.draw_lists_and_children);
    }

    /// Calcula el daño del frame, en píxeles de dispositivo.
    ///
    /// Devuelve `None` cuando hay que repintar entero: o porque se pidió
    /// explícitamente, o —y esto importa igual— porque **no se sabe**. Un daño
    /// desconocido no es un daño vacío: es la pantalla entera. Confundir los dos
    /// es cómo se llega a una pantalla que no se refresca y nadie sabe por qué.
    pub fn calcular(&mut self, cx: &Cx, dpi: f64, width: i32, height: i32) -> Option<DamageRect> {
        if self.todo {
            return None;
        }
        if self.pendientes.is_empty() && self.pendientes_con_hijos.is_empty() {
            // Nada se ensució y aun así estamos pintando: no lo sabemos → entero.
            return None;
        }

        let mut listas: Vec<DrawListId> = self.pendientes.clone();
        for id in &self.pendientes_con_hijos {
            recoger_con_hijos(cx, *id, &mut listas, 0);
        }

        let mut dano: Option<DamageRect> = None;
        for id in &listas {
            // Lo de AHORA.
            if let Some(r) = rect_de_lista(cx, *id, dpi) {
                dano = Some(match dano {
                    Some(d) => d.union(r),
                    None => r,
                });
                self.anterior.insert(*id, r);
            }
            // Y lo de ANTES, para no dejar fantasmas donde estaba.
            if let Some(prev) = self.anterior.get(id).copied() {
                dano = Some(match dano {
                    Some(d) => d.union(prev),
                    None => prev,
                });
            }
        }

        if std::env::var("MAKEPAD_HEADLESS_DAMAGE_TRACE").is_ok() {
            for id in &listas {
                crate::log!("[damage]   lista {:?}: {:?}", id, rect_de_lista(cx, *id, dpi));
            }
        }
        let dano = dano?.clamp(width, height);
        if dano.is_empty() {
            return None;
        }
        // Si el daño cubre casi todo, el scissor sólo añade trabajo. El umbral es
        // deliberadamente alto (90 %): por debajo de eso el recorte sigue pagando.
        if dano.area() * 10 >= (width as u64) * (height as u64) * 9 {
            return None;
        }
        Some(dano)
    }
}

/// Añade `id` y, recursivamente, sus sub-listas.
///
/// La profundidad se limita porque el árbol de draw lists lo construyen los
/// widgets y un ciclo (una lista que se contiene a sí misma por un fallo de
/// reciclado de ids) colgaría el render sin decir por qué. 64 niveles es muy por
/// encima de cualquier UI real.
fn recoger_con_hijos(cx: &Cx, id: DrawListId, salida: &mut Vec<DrawListId>, prof: u32) {
    if prof > 64 || salida.contains(&id) {
        return;
    }
    salida.push(id);
    let Some(lista) = cx.draw_lists.checked_index(id) else {
        return;
    };
    // `CxDrawItems` indexa por posición y no expone iterador: se recorre por
    // índice, que es como lo hace el propio makepad al pintar.
    let hijos: Vec<DrawListId> = (0..lista.draw_items.len())
        .filter_map(|i| lista.draw_items[i].kind.sub_list())
        .collect();
    for hijo in hijos {
        recoger_con_hijos(cx, hijo, salida, prof + 1);
    }
}

/// Caja de una draw list, en píxeles de dispositivo.
///
/// Sale de `painted_rect` (ver la cabecera del módulo: `rect_areas` NO sirve para
/// esto). Se redondea hacia fuera —`floor` abajo, `ceil` arriba— a propósito: un
/// píxel de más se repinta sin que se note, uno de menos deja un borde viejo.
/// Cuando hay que equivocarse, se hace hacia el lado que no se ve.
fn rect_de_lista(cx: &Cx, id: DrawListId, dpi: f64) -> Option<DamageRect> {
    let lista = cx.draw_lists.checked_index(id)?;
    let r = lista.painted_rect?;
    let d = DamageRect {
        x0: (r.pos.x * dpi).floor() as i32,
        y0: (r.pos.y * dpi).floor() as i32,
        x1: ((r.pos.x + r.size.x) * dpi).ceil() as i32,
        y1: ((r.pos.y + r.size.y) * dpi).ceil() as i32,
    };
    if d.is_empty() {
        return None;
    }
    Some(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(x0: i32, y0: i32, x1: i32, y1: i32) -> DamageRect {
        DamageRect { x0, y0, x1, y1 }
    }

    #[test]
    fn la_union_contiene_a_los_dos() {
        let a = r(10, 10, 20, 20);
        let b = r(30, 5, 40, 15);
        let u = a.union(b);
        assert_eq!(u, r(10, 5, 40, 20));
        // Y contiene a los dos, que es la propiedad que importa.
        for x in [a, b] {
            assert!(u.x0 <= x.x0 && u.y0 <= x.y0 && u.x1 >= x.x1 && u.y1 >= x.y1);
        }
    }

    #[test]
    fn la_union_es_conmutativa_y_absorbe_lo_contenido() {
        let a = r(0, 0, 100, 100);
        let b = r(10, 10, 20, 20);
        assert_eq!(a.union(b), b.union(a));
        assert_eq!(a.union(b), a);
    }

    #[test]
    fn un_rectangulo_degenerado_esta_vacio_y_no_tiene_area() {
        for d in [r(5, 5, 5, 10), r(5, 5, 10, 5), r(10, 10, 5, 5)] {
            assert!(d.is_empty());
            assert_eq!(d.area(), 0);
        }
    }

    #[test]
    fn el_recorte_a_pantalla_no_deja_coordenadas_fuera() {
        let d = r(-50, -50, 2000, 2000).clamp(1280, 800);
        assert_eq!(d, r(0, 0, 1280, 800));
        assert_eq!(d.area(), 1280 * 800);
    }

    /// El daño de un rectángulo completamente fuera de la pantalla se queda vacío,
    /// no negativo. Un área negativa desbordaría el `u64` del contador.
    #[test]
    fn lo_que_cae_fuera_de_la_pantalla_no_da_area_negativa() {
        let d = r(2000, 2000, 3000, 3000).clamp(1280, 800);
        assert!(d.is_empty());
        assert_eq!(d.area(), 0);
    }
}
