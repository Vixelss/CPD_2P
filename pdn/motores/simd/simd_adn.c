/*
 * simd_adn.c - nucleo de calculo en C con intrinsecas AVX2 explicitas.
 *
 * El mismo archivo se compila dos veces (scripts/compilar_simd.sh):
 *   libsimd_avx2.so    : gcc -O3 -mavx2 -mpopcnt      -> rama __AVX2__
 *   libsimd_escalar.so : gcc -O3 -fno-tree-vectorize  -> rama escalar
 * Asi la comparacion escalar contra AVX2 es justa: mismo codigo, mismas
 * funciones exportadas, y el compilador no vectoriza la version escalar.
 *
 * Funciones exportadas (llamadas desde Python con ctypes):
 *   contar_simbolos_avx2  cuenta los 30 bytes validos (ACGTN + IUPAC, en
 *                         mayuscula y minuscula); devuelve los invalidos.
 *   histograma_escalar    histograma de 256 casillas con 4 tablas parciales.
 *   comparar_avx2         diferencias por categoria (solo caso, con N, reales).
 *   buscar_patron_avx2    posiciones donde empieza un patron.
 *   es_avx2               1 si la biblioteca se compilo con AVX2.
 *
 * Comentarios sin tildes (convencion del proyecto).
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* Los 30 simbolos validos en el orden del arreglo de salida */
static const uint8_t SIMBOLOS[30] = {
    'A', 'C', 'G', 'T', 'N', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V',
    'a', 'c', 'g', 't', 'n', 'r', 'y', 's', 'w', 'k', 'm', 'b', 'd', 'h', 'v'};

int es_avx2(void) {
#ifdef __AVX2__
    return 1;
#else
    return 0;
#endif
}

/* Histograma escalar con 4 tablas parciales: rompe la dependencia de
 * memoria cuando bytes consecutivos caen en la misma casilla. El histograma
 * completo no se vectoriza de forma directa (no hay scatter con suma en
 * AVX2), por eso esta funcion es escalar en las dos bibliotecas. */
void histograma_escalar(const uint8_t *d, size_t n, uint64_t *hist) {
    uint64_t t0[256], t1[256], t2[256], t3[256];
    memset(t0, 0, sizeof t0);
    memset(t1, 0, sizeof t1);
    memset(t2, 0, sizeof t2);
    memset(t3, 0, sizeof t3);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        t0[d[i]]++;
        t1[d[i + 1]]++;
        t2[d[i + 2]]++;
        t3[d[i + 3]]++;
    }
    for (; i < n; i++) t0[d[i]]++;
    for (int k = 0; k < 256; k++) hist[k] += t0[k] + t1[k] + t2[k] + t3[k];
}

/* Cuenta los 30 simbolos validos. salida[30] se acumula (no se pone a
 * cero). Devuelve el numero de bytes invalidos del tramo (los que no son
 * ninguno de los 30). */
uint64_t contar_simbolos_avx2(const uint8_t *d, size_t n, uint64_t *salida) {
    size_t i = 0;
    uint64_t antes = 0;
    for (int k = 0; k < 30; k++) antes += salida[k];
#ifdef __AVX2__
    /* Por cada simbolo: cmpeq da -1 en los bytes que coinciden; restar ese
     * -1 a un acumulador de 32 contadores de 8 bits suma 1. Cada 255 vueltas
     * se vacia con _mm256_sad_epu8 a contadores de 64 bits antes de que se
     * desborden. Se procesan 10 simbolos por pasada para que los
     * acumuladores quepan en los 16 registros ymm, y el bloque de 255 x 32
     * bytes (8 KB) se queda en la cache L1 entre las tres pasadas. */
    const __m256i cero = _mm256_setzero_si256();
    while (i + 32 <= n) {
        size_t vueltas = (n - i) / 32;
        if (vueltas > 255) vueltas = 255;
        for (int grupo = 0; grupo < 3; grupo++) {
            __m256i ref[10], acc[10];
            for (int k = 0; k < 10; k++) {
                ref[k] = _mm256_set1_epi8((char)SIMBOLOS[grupo * 10 + k]);
                acc[k] = cero;
            }
            const uint8_t *p = d + i;
            for (size_t v = 0; v < vueltas; v++, p += 32) {
                __m256i x = _mm256_loadu_si256((const __m256i *)p);
                for (int k = 0; k < 10; k++)
                    acc[k] = _mm256_sub_epi8(acc[k], _mm256_cmpeq_epi8(x, ref[k]));
            }
            for (int k = 0; k < 10; k++) {
                __m256i s = _mm256_sad_epu8(acc[k], cero);
                salida[grupo * 10 + k] += (uint64_t)_mm256_extract_epi64(s, 0) +
                                          (uint64_t)_mm256_extract_epi64(s, 1) +
                                          (uint64_t)_mm256_extract_epi64(s, 2) +
                                          (uint64_t)_mm256_extract_epi64(s, 3);
            }
        }
        i += vueltas * 32;
    }
#endif
    /* Resto escalar (o todo, en la biblioteca escalar) */
    int8_t pos[256];
    memset(pos, -1, sizeof pos);
    for (int k = 0; k < 30; k++) pos[SIMBOLOS[k]] = (int8_t)k;
    for (; i < n; i++) {
        int k = pos[d[i]];
        if (k >= 0) salida[k]++;
    }
    uint64_t despues = 0;
    for (int k = 0; k < 30; k++) despues += salida[k];
    return (uint64_t)n - (despues - antes); /* invalidos del tramo */
}

/* Diferencias entre a y b por categoria. categorias[3] se acumula:
 * [0] solo caso (a != b y a&0xDF == b&0xDF), [1] con N, [2] reales.
 * Devuelve el total de diferencias del tramo. */
uint64_t comparar_avx2(const uint8_t *a, const uint8_t *b, size_t n, uint64_t *categorias) {
    uint64_t solo = 0, con_n = 0, reales = 0;
    size_t i = 0;
#ifdef __AVX2__
    const __m256i mascara = _mm256_set1_epi8((char)0xDF);
    const __m256i letra_n = _mm256_set1_epi8('N');
    for (; i + 32 <= n; i += 32) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i y = _mm256_loadu_si256((const __m256i *)(b + i));
        uint32_t igual = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, y));
        if (igual == 0xFFFFFFFFu) continue; /* caso comun: 32 bytes iguales */
        __m256i ux = _mm256_and_si256(x, mascara);
        __m256i uy = _mm256_and_si256(y, mascara);
        uint32_t igual_may = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(ux, uy));
        uint32_t hay_n = (uint32_t)_mm256_movemask_epi8(_mm256_or_si256(
            _mm256_cmpeq_epi8(ux, letra_n), _mm256_cmpeq_epi8(uy, letra_n)));
        uint32_t dif = ~igual;
        uint32_t m_solo = dif & igual_may;
        uint32_t resto = dif & ~igual_may;
        solo += (uint64_t)__builtin_popcount(m_solo);
        con_n += (uint64_t)__builtin_popcount(resto & hay_n);
        reales += (uint64_t)__builtin_popcount(resto & ~hay_n);
    }
#endif
    for (; i < n; i++) {
        if (a[i] == b[i]) continue;
        uint8_t ua = a[i] & 0xDF, ub = b[i] & 0xDF;
        if (ua == ub)
            solo++;
        else if (ua == 'N' || ub == 'N')
            con_n++;
        else
            reales++;
    }
    categorias[0] += solo;
    categorias[1] += con_n;
    categorias[2] += reales;
    return solo + con_n + reales;
}

/* Comprueba si el patron p (en mayuscula, N = comodin de A C G T) empieza
 * en d (sin distinguir caso). Con m = 0 coincide siempre. */
static inline int coincide_en(const uint8_t *d, const uint8_t *p, size_t m) {
    for (size_t k = 0; k < m; k++) {
        uint8_t u = d[k] & 0xDF;
        if (p[k] == 'N') {
            if (u != 'A' && u != 'C' && u != 'G' && u != 'T') return 0;
        } else if (u != p[k]) {
            return 0;
        }
    }
    return 1;
}

/* Busca el patron p de largo m en d[0, n). Escribe en pos las posiciones de
 * inicio (hasta max_pos) y devuelve cuantas coincidencias hubo. Compara el
 * primer byte del patron (y el segundo) en paralelo de 32 en 32 y verifica
 * los candidatos. */
uint64_t buscar_patron_avx2(const uint8_t *d, size_t n, const uint8_t *p, size_t m,
                            uint64_t *pos, uint64_t max_pos) {
    uint64_t encontrados = 0;
    if (m == 0 || n < m) return 0;
    size_t ultimo = n - m; /* ultimo inicio posible */
    size_t i = 0;
#ifdef __AVX2__
    if (m >= 2 && p[0] != 'N' && p[1] != 'N') {
        /* Filtro con los dos primeros bytes: se compara d[i..i+31] con p[0] y
         * d[i+1..i+32] con p[1]; solo las posiciones que cumplen ambas son
         * candidatas (en ADN, una de cada ~16 en lugar de una de cada 4). */
        const __m256i mascara = _mm256_set1_epi8((char)0xDF);
        const __m256i primero = _mm256_set1_epi8((char)p[0]);
        const __m256i segundo = _mm256_set1_epi8((char)p[1]);
        for (; i + 32 <= ultimo + 1; i += 32) {
            __m256i x0 = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)(d + i)), mascara);
            __m256i x1 = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)(d + i + 1)), mascara);
            __m256i e = _mm256_and_si256(_mm256_cmpeq_epi8(x0, primero), _mm256_cmpeq_epi8(x1, segundo));
            uint32_t cand = (uint32_t)_mm256_movemask_epi8(e);
            while (cand) {
                unsigned k = (unsigned)__builtin_ctz(cand);
                cand &= cand - 1;
                if (coincide_en(d + i + k + 2, p + 2, m - 2)) {
                    if (encontrados < max_pos) pos[encontrados] = i + k;
                    encontrados++;
                }
            }
        }
    }
#endif
    for (; i <= ultimo; i++) {
        if (coincide_en(d + i, p, m)) {
            if (encontrados < max_pos) pos[encontrados] = i;
            encontrados++;
        }
    }
    return encontrados;
}
