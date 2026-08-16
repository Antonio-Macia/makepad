#!/usr/bin/env python3
"""Compara dos series de frames: con daño calculado y con repintado completo.

Lo llama `verificar-dano.sh`. Va en un fichero propio y no en un heredoc dentro
del `.sh` porque el script de Python contiene su propio bloque `PY`, y anidar
heredocs termina el de fuera antes de tiempo — un rato perdido el 2026-08-16.

Códigos de salida: 0 verificado · 1 roto · 75 no verificado (los TRES estados;
fundir el 75 con el 0 es cómo un baseline roto sobrevive días).
"""

import glob
import os
import sys

try:
    from PIL import Image, ImageChops
except ImportError:
    print("NO VERIFICADO: hace falta Pillow (python3 -m pip install pillow).")
    sys.exit(75)


def abrir(ruta):
    return Image.open(ruta).convert("RGB")


def main():
    if len(sys.argv) != 3:
        print("uso: comparar-dano.py <dir_con_dano> <dir_referencia>")
        return 75

    con = sorted(glob.glob(os.path.join(sys.argv[1], "*.png")))
    ref = sorted(glob.glob(os.path.join(sys.argv[2], "*.png")))

    if not con or not ref:
        print("NO VERIFICADO: no se generó ningún frame.")
        return 75
    if len(con) != len(ref):
        print(f"NO VERIFICADO: {len(con)} frames con daño y {len(ref)} sin él "
              "— no son comparables.")
        return 75
    if len(con) < 3:
        print(f"NO VERIFICADO: hacen falta al menos 3 frames y hay {len(con)}.")
        return 75

    # ── Paso 1: ¿es la REFERENCIA estable consigo misma? ─────────────────────
    #
    # No se le puede exigir al daño una estabilidad que la propia vara de medir
    # no tiene. Y aquí no es hipotético: el backend por software pinta la
    # primera fila de texto más APAGADA en el frame 0 que en todos los
    # siguientes. Es un defecto PRE-EXISTENTE — confirmado el 2026-08-16 contra
    # el código anterior a la persistencia y al cálculo de daño. Con repintado
    # completo se autocorrige en el frame 1 y nadie lo nota; con daño, esa zona
    # se congela porque nadie la vuelve a tocar.
    #
    # Se mide la referencia contra sí misma, se saca la máscara de lo que ella
    # no reproduce, y se excluye. Y se IMPRIME siempre: una exclusión callada es
    # cómo un oráculo se convierte en una casilla.
    r0, r1 = abrir(ref[0]), abrir(ref[1])
    inestable = (
        ImageChops.difference(r0, r1).convert("L").point(lambda v: 255 if v else 0)
    )
    zona = inestable.getbbox()
    px_inestables = sum(1 for v in inestable.getdata() if v)

    if zona:
        x0, y0, x1, y1 = zona
        print(f"⚠  la REFERENCIA no se reproduce a sí misma en {px_inestables} px, "
              f"en ({x0},{y0})-({x1},{y1}).")
        print("   Es el defecto pre-existente del frame 0 (texto más apagado).")
        print("   Esa zona se EXCLUYE: el daño no puede responder de ella.")

    # ── Paso 2: comparar, desde el frame 1 y fuera de la zona inestable ──────
    #
    # Desde el frame 1 porque el frame 0 es repintado completo en las DOS
    # corridas por construcción (no hay frame anterior que conservar), así que
    # compararlo no prueba nada.
    malos = []
    for i in range(1, len(con)):
        ia, ib = abrir(con[i]), abrir(ref[i])
        if ia.size != ib.size:
            print(f"NO VERIFICADO: {os.path.basename(con[i])} mide {ia.size} "
                  f"y su pareja {ib.size}.")
            return 75
        d = ImageChops.difference(ia, ib).convert("L")
        d = ImageChops.subtract(d, inestable)
        if d.getbbox() is None:
            continue
        n = sum(1 for v in d.getdata() if v)
        if n:
            malos.append((os.path.basename(con[i]), n, d.getbbox()))

    comparados = len(con) - 1
    if not malos:
        print(f"✅ VERIFICADO: {comparados} frames comparados, 0 px de diferencia "
              f"(excluidos {px_inestables} px que la referencia no reproduce).")
        return 0

    print(f"🔴 ROTO: {len(malos)} de {comparados} frames difieren.")
    print("   El rectángulo de cada línea es lo que el daño se dejó fuera.")
    for nombre, n, bbox in malos[:12]:
        x0, y0, x1, y1 = bbox
        print(f"   {nombre}  {n} px  en ({x0},{y0})-({x1},{y1})")
    if len(malos) > 12:
        print(f"   … y {len(malos) - 12} más")
    return 1


if __name__ == "__main__":
    sys.exit(main())
