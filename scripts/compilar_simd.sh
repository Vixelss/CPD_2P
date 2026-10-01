#!/usr/bin/env bash
# Compila el nucleo SIMD en dos bibliotecas: AVX2 y escalar (sin vectorizar).
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUENTE="$RAIZ/pdn/motores/simd/simd_adn.c"
SALIDA="$RAIZ/pdn/motores/simd/build"
CC="${CC:-gcc}"

if ! command -v "$CC" >/dev/null 2>&1; then
    echo "ERROR: no se encontro el compilador '$CC'. Instale build-essential." >&2
    exit 1
fi
mkdir -p "$SALIDA"

"$CC" -O3 -fno-tree-vectorize -shared -fPIC -o "$SALIDA/libsimd_escalar.so" "$FUENTE"
echo "Compilada $SALIDA/libsimd_escalar.so"

if grep -qw avx2 /proc/cpuinfo 2>/dev/null; then
    "$CC" -O3 -mavx2 -mpopcnt -shared -fPIC -o "$SALIDA/libsimd_avx2.so" "$FUENTE"
    echo "Compilada $SALIDA/libsimd_avx2.so"
else
    echo "AVISO: este procesador no tiene AVX2; solo se compila la version escalar."
fi
