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
| E3 Master, workers y clúster simulado | Hecha |
| E4 Modo MPI y escalabilidad | Hecha |
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

### E3. Master, workers y clúster simulado

- `pdn/comun/config.py` (cluster.yaml + valores por defecto), `protocolo.py` (mensajes JSON; se agregaron `PREPARAR` y `LISTO`), `registro_log.py` (log a consola, archivo y memoria para el dashboard).
- `pdn/master/`: `planificador.py` (adaptativa con ración mínima inicial y fase final decreciente, fija, proporcional, preferencia NPU>GPU>CPU en zonas), `estado.py`, `servidor.py` (`Master`: hilo ZeroMQ ROUTER, API segura entre hilos: `iniciar_corrida`, `esperar_corrida`, `correr`, `simular_fallo`, `cancelar_corrida`, `estado`), `validacion.py` (CRC por unidad, cobertura, referencia), `resultados.py` (resumen.json, resultado.json, tareas.csv, config.json, recursos.csv).
- `pdn/worker/`: `worker.py` (bucle petición-respuesta, hilo de latidos, fallos simulados `caida` y `congelado`, rotación entre Masters si no hay respuesta en 3 s), `motores.py` (fábrica por dispositivo), `__main__.py`.
- `pdn/monitoreo/recursos.py`: `Monitor` básico (se completa en E6).
- `pdn/sim/cluster.py`: `ClusterSimulado` (Master local + workers en procesos con retardo por MB, copias dañadas a propósito).
- `pdn/cli.py`: `preparar`, `master`, `sim`, `correr` (vía API del dashboard, E7), `referencia` (E4).
- Pruebas: `tests/test_planificador.py`, `tests/test_cluster_simulado.py` (resultado exacto de las 4 operaciones distribuidas; el nodo rápido procesa más y el ocioso final es < 0,5 s; 20 combinaciones de unidad y tiempo objetivo con 3 y 5 workers; caída y congelado a mitad de corrida; SIGKILL; copia con un byte cambiado rechazada por CRC y nodo marcado sospechoso; copia con huella global distinta excluida; persistencia; errores de configuración). Se corrieron 3 veces seguidas sin fallos.
- Demo: `python -m pdn.cli sim --workers 4 --retardos 0,0.2,0.5,1 --operacion patrones --mb 4`.

### E4. Modo MPI, referencia secuencial y escalabilidad

- `pdn/mpi/reparto.py` (iguales o proporcional), `mpi_correr.py` (programa por rank: `scatter` del reparto, `Barrier` + `MPI.Wtime`, `Reduce` con `MPI.SUM` del histograma, `gather` del resto, validación contra la referencia), `lanzador.py` (`mpirun --map-by core --bind-to core --report-bindings --mca btl_tcp_if_include ...`, guarda `bindings.txt` como evidencia), `referencia.py` (T₁ con un núcleo), `escalabilidad.py` (series 1 a 5, CSV, `amdahl.json`, `escalabilidad.png`).
- `scripts/generar_hostfile.py` (lee `slots` de cada nodo en `cluster.yaml`; `--max-slots 1` y `--nodos N` para las series).
- CLI: `python -m pdn.cli referencia ...`, `python -m pdn.cli correr --modo mpi --np 8 --hostfile hostfile [--reparto proporcional]`, `python -m pdn.mpi.escalabilidad --archivo X [--solo-local]`.
- Se corrigió `cluster.yaml`: en E0 había quedado con las marcas de bloque de código de Markdown y no se podía leer. Hay una prueba que lo verifica.
- Hallazgo de rendimiento (ver decisión 26): las operaciones ahora procesan por subtramos para acotar la memoria temporal de numpy.
- En la nube (OpenMPI 4.1.6 instalado con apt, `mpi4py` con pip, 4 núcleos, 64 MiB sintéticos, mediana de 3 con calentamiento): conteo T₁ = 0,36 s, speedup 1,89 con 2 y 3,42 con 4 (s de Amdahl = 0,057); patrones 3,48 con 4; zonas 3,52 con 4 (s = 0,043).
- Pruebas: `tests/test_mpi.py` (reparto, ajuste de Amdahl, referencia + validación del Master, `mpirun -np 1/2/4` local igual a la referencia en conteo, patrones y zonas, comparación con SIMD, reparto proporcional, serie de escalabilidad con CSV y PNG, comando del lanzador).

## Pendiente de prueba en hardware

- Afinidad con núcleos P y E reales en el i5-13420H de `nodo-vivanco` (en la nube no hay CPU híbrida; la lógica se probó con una topología simulada).
- Benchmark SIMD en cada laptop (`python -m herramientas.benchmark_simd --seq ~/pdn-datos/GCF_000001405.40_GRCh38.p14_genomic.seq`).
- MPI multinodo con el hostfile real: `scripts/generar_hostfile.py` y luego `python -m pdn.mpi.escalabilidad --archivo GCF_000001405.40_GRCh38.p14_genomic --repeticiones 3 --calentamiento` (series 3 y 4 solo se pueden correr con las laptops).
- Valores de aceptación de las secciones 8.1 y 8.3 sobre los genomas reales (`GCF_000001405.40` y `GCA_000001405.29`): conteo exacto, 701 parejas, 0 diferencias emparejado, 1.072.801.765 posicional.

## Preguntas abiertas para el profesor

1. ¿El tiempo de distribuir el archivo a los nodos cuenta en el examen?
2. ¿Acepta Core ML con el Neural Engine de Apple como implementación de NPU, en lugar de OpenVINO o TensorRT?
3. Con el archivo de 10 GB del examen, ¿qué operación se pide exactamente?
