# ESTADO DEL PROYECTO

Traspaso entre sesiones de Claude Code. Debe bastar para retomar sin leer el historial. La especificación es `CONTEXTO.md`.

## Entorno de desarrollo usado en la nube

- Python 3.12 en `.venv/` (ignorado por Git): `python3.12 -m venv .venv && .venv/bin/pip install -r requirements/base.txt numba`.
- Pruebas: `.venv/bin/python -m pytest -q`.

## Etapas

| Etapa | Estado |
|---|---|
| E0 Reorganización y esqueleto | Hecha |
| E1 Datos y operaciones de referencia | Hecha |
| E2 Motor CPU y núcleo SIMD | Hecha |
| E3 Master, workers y clúster simulado | Pendiente |
| E4 Modo MPI y escalabilidad | Pendiente |
| E5 Motor GPU CUDA | Pendiente |
| E6 Monitoreo y energía | Pendiente |
| E7 Dashboard | Pendiente |
| E8 Alta disponibilidad | Pendiente |
| E9 NPU en la Mac | Pendiente |
| E10 Despliegue y manual | Pendiente |
| E11 Opcionales | Pendiente |

### E0. Reorganización y esqueleto

- Hecho: `P1.1`–`P1.4` movidos a `referencias/` con `git mv`; una sola copia del enunciado en `referencias/enunciado.pdf`; borrado `referencias/P1.2/__pycache__/`; `.gitignore` y `.gitattributes` (finales de línea normalizados a LF); estructura de carpetas de la sección 15; `requirements/`; `cluster.yaml` de ejemplo; `pytest.ini` con marcadores `gpu`, `npu`, `multinodo`, `lento`.
- No había `vendor/`, `datos/` ni archivos `.fna`/`.seq`/`.idx` versionados.
- Pruebas: `pytest` corre (1 prueba de humo).

### E1. Datos y operaciones de referencia

- `pdn/comun/`: `formato.py` (clases de byte, `resumir_histograma`, fila y columna), `unidades.py` (unidad de 4 MiB), `huellas.py` (`.origen` y CRC32 por unidad).
- `pdn/preparacion/`: `fasta_a_seq.py` (`preparar`: `.seq`, `.idx`, `.huellas`, `.origen` en una pasada, cache validada por huella), `indice.py` (`Indice.localizar`, `limites_para`), `emparejar.py`.
- `pdn/operaciones/`: `conteo`, `patrones`, `comparacion`, `zonas`. Interfaz común por módulo: `validar_parametros`, `solape`, `procesar`, `vacio`, `combinar`, `finalizar`, `clave_comparable`. `trabajo.py` define `Trabajo` (operación + índices + espacio a repartir, `carga(inicio, fin)` para una tarea) y `ejecutar_secuencial`.
- `herramientas/`: `generar_sintetico.py` (FASTA con respuesta conocida, variante CRLF, pares con sustituciones y par reordenado), `ensuciar.py`, `generar_par.py`, `recortar.py`, `fasta_crudo.py`.
- Pruebas: `tests/test_preparacion.py`, `tests/test_operaciones.py` (12 combinaciones de tamaño de unidad y de tarea para las costuras), `tests/test_herramientas.py`. Se comprobó que las pruebas fallan si se rompe el manejo de límites de registro o el solape.
- Interfaz de una tarea: `{"inicio", "fin", "solape", "limites": [[indice_registro, inicio], ...]}`; en comparación `{"segmentos": [[id_pareja, ini_a, ini_b, largo], ...]}`.

### E2. Motor CPU y núcleo SIMD

- `pdn/operaciones/nucleo.py`: interfaz de núcleo (`NucleoNumpy`); las operaciones aceptan `nucleo=`.
- `pdn/motores/simd/simd_adn.c` + `envoltorio.py` (ctypes): `contar_simbolos_avx2`, `histograma_escalar`, `comparar_avx2`, `buscar_patron_avx2`, `es_avx2`. `scripts/compilar_simd.sh` compila `build/libsimd_avx2.so` y `build/libsimd_escalar.so` (el envoltorio compila solo si faltan).
- `pdn/motores/cpu.py`: `MotorCPU(procesos, nucleos, impl, reservar)`; pool persistente, afinidad por proceso con `sched_setaffinity`, `partir_carga`, `describir()` con la asignación núcleo↔pid.
- `pdn/worker/hardware.py`: detección completa (CPU, núcleos P/E, hermanos HT, RAM, GPU CUDA, NPU, OpenCL, red, OpenMPI) y `elegir_nucleos` con validación.
- `herramientas/benchmark_simd.py` escribe `resultados/benchmark_simd.csv`. En la nube (Xeon 2,1 GHz, 32 MB, mediana de 3): histograma numpy 232 MB/s, C escalar 1003, AVX2 2061; comparación numpy 211, escalar 2282, AVX2 12566; patrón GATTACA numpy 497, escalar 175, AVX2 1070.
- Pruebas: `tests/test_motor_cpu.py` (equivalencia C contra numpy con longitudes no múltiplo de 32, motor con 1 y N procesos, numpy y SIMD, 4 tamaños de unidad, afinidad registrada, validaciones de rango).

## Pendiente de prueba en hardware

- Afinidad con núcleos P y E reales en el i5-13420H de `nodo-vivanco` (en la nube no hay CPU híbrida; la lógica se probó con una topología simulada).
- Benchmark SIMD en cada laptop (`python -m herramientas.benchmark_simd --seq ~/pdn-datos/GCF_000001405.40_GRCh38.p14_genomic.seq`).
- Valores de aceptación de las secciones 8.1 y 8.3 sobre los genomas reales (`GCF_000001405.40` y `GCA_000001405.29`): conteo exacto, 701 parejas, 0 diferencias emparejado, 1.072.801.765 posicional.

## Preguntas abiertas para el profesor

1. ¿El tiempo de distribuir el archivo a los nodos cuenta en el examen?
2. ¿Acepta Core ML con el Neural Engine de Apple como implementación de NPU, en lugar de OpenVINO o TensorRT?
3. Con el archivo de 10 GB del examen, ¿qué operación se pide exactamente?
