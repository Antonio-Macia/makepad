#!/usr/bin/env bash
# Oráculo del repintado parcial: ¿el daño calculado se deja algo fuera?
#
# ─ QUÉ COMPRUEBA ─────────────────────────────────────────────────────────────
#
# Corre la MISMA escena dos veces:
#   1. con el daño calculado (repinta sólo el rectángulo sucio),
#   2. sin él (repinta la pantalla entera cada frame).
#
# Si el daño es correcto, las dos series de imágenes son idénticas píxel a
# píxel. Cualquier diferencia es, literalmente, lo que el daño se dejó fuera.
#
# ─ POR QUÉ ESTE ORÁCULO Y NO OTRO ────────────────────────────────────────────
#
# El fallo del repintado parcial NO da error: la aplicación arranca, compila y
# la captura sale bien. Lo que deja son restos del frame anterior donde algo se
# encogió o se movió — y eso sólo aparece EN MOVIMIENTO, nunca en una captura
# estática, que es justo lo que se suele mirar.
#
# Y el oráculo no sale del mismo sitio que la implementación: la referencia es
# el camino de pantalla completa, que no comparte ni una línea con el cálculo
# del daño. Si los dos estuvieran mal de la misma manera, coincidirían — pero
# el camino completo no tiene nada que calcular.
#
# ─ CÓDIGOS DE SALIDA ─────────────────────────────────────────────────────────
#
#   0  verificado: 0 píxeles de diferencia en todos los frames
#   1  ROTO: algún frame difiere (se imprime cuál y su rectángulo)
#  75  NO VERIFICADO: no se pudo ejecutar (falta el binario, falta Python/PIL,
#      no salieron frames). NO es un aprobado: es la tercera respuesta, y
#      fundirla con el 0 es cómo un baseline roto sobrevive días.
#
# ─ USO ───────────────────────────────────────────────────────────────────────
#
#   tools/verificar-dano.sh [frames]      # por defecto 14

set -uo pipefail

FRAMES="${1:-14}"
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$RAIZ/target/release/makepad-example-damage-bench"
CON="${TMPDIR:-/tmp}/dano_con"
SIN="${TMPDIR:-/tmp}/dano_sin"

if [[ ! -x "$BIN" ]]; then
  echo "NO VERIFICADO: falta el binario del banco."
  echo "  MAKEPAD=headless cargo build -p makepad-example-damage-bench --release"
  exit 75
fi

if ! python3 -c "import PIL" 2>/dev/null; then
  echo "NO VERIFICADO: hace falta Pillow (python3 -m pip install pillow)."
  exit 75
fi

# ⚠ El binario tiene que ser el ACTUAL, y esto se comprueba ANTES de correr nada.
# Si el build falló, `target/` conserva el anterior y la comparación mide, tan
# tranquila, la versión de antes — con un resultado perfectamente creíble y
# falso. Pasó dos veces el mismo día (2026-08-16): una por compilar desde el
# workspace equivocado y otra porque un `git stash` se llevó el `Cargo.toml` que
# daba de alta el banco. Las dos veces el síntoma fue el mismo: cifras IDÉNTICAS
# a la corrida anterior.
for fuente in platform/src/os/headless/damage.rs \
              platform/src/os/headless/raster.rs \
              platform/src/draw_list.rs \
              draw/src/cx_2d.rs \
              examples/damage_bench/src/main.rs; do
  if [[ -f "$RAIZ/$fuente" && "$BIN" -ot "$RAIZ/$fuente" ]]; then
    echo "NO VERIFICADO: el binario es MÁS VIEJO que $fuente."
    echo "  MAKEPAD=headless cargo build -p makepad-example-damage-bench --release"
    exit 75
  fi
done

rm -rf "$CON" "$SIN"; mkdir -p "$CON" "$SIN"

export MAKEPAD=headless
export MAKEPAD_HEADLESS_DPI=1

echo "── corrida 1/2: con daño calculado ($FRAMES frames)"
MAKEPAD_HEADLESS_DAMAGE=1 MAKEPAD_HEADLESS_OUT_DIR="$CON" \
  timeout 600 "$BIN" --draws="$FRAMES" >/dev/null 2>&1

echo "── corrida 2/2: pantalla entera (referencia)"
MAKEPAD_HEADLESS_OUT_DIR="$SIN" \
  timeout 600 "$BIN" --draws="$FRAMES" >/dev/null 2>&1

exec python3 "$RAIZ/tools/comparar-dano.py" "$CON" "$SIN"
