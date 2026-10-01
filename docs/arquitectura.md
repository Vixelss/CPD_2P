# Arquitectura

## Vista de conjunto

```
                         ┌──────────────────────────────────────────┐
                         │ nodo-vivanco (Master)                    │
                         │  Master (hilo ZeroMQ ROUTER :5555)       │
                         │   ├─ Planificador (raciones adaptativas) │
                         │   ├─ Registro de tareas y reasignación   │
                         │   ├─ Validación (CRC, cobertura, ref.)   │
                         │   └─ Replicador ──────────┐ :5556        │
                         │  Dashboard FastAPI :8000  │              │
                         │  NFS /cluster             │              │
                         │  worker cpu + worker gpu  │              │
                         └──────────────┬────────────┼──────────────┘
                                        │            ▼
                    ZeroMQ DEALER       │   nodo-carranza: MasterRespaldo
                    (petición-respuesta │   (se promueve si pasan 3 s
                     + latidos)         │    sin instantáneas)
     ┌──────────────┬───────────────────┼──────────────┬──────────────┐
 nodo-carranza  nodo-ocampo       nodo-naranjo     nodo-palomo    nodo-hidalgo
 worker cpu     worker cpu        worker cpu+gpu   worker cpu     worker npu (Core ML)
```

## Capas del código

| Capa | Paquete | Qué hace |
|---|---|---|
| Datos | `pdn/preparacion/` | `.fna` → `.seq` (secuencia limpia), `.idx` (registros), `.huellas` (CRC32 por unidad), `.origen` (huella del archivo de origen); emparejamiento GenBank/RefSeq |
| Definición de las operaciones | `pdn/operaciones/` | Conteo, patrones, comparación y zonas en un hilo: costuras, límites de registro, consolidación. `Trabajo` traduce un rango del espacio a la carga de una tarea |
| Núcleos de cálculo | `pdn/operaciones/nucleo.py`, `pdn/motores/simd/`, `pdn/motores/gpu_cuda.py`, `pdn/motores/npu/` | Histograma, coincidencias, categorías de diferencia, conteo por ventanas y clasificación: numpy, C (escalar y AVX2), CUDA, Core ML |
| Motores | `pdn/motores/cpu.py`, `gpu_cuda.py`, `npu/motor_npu.py` | Ejecutan una tarea en un dispositivo: pool con afinidad, tubo con dos streams, modelo cuantizado |
| Modo dinámico | `pdn/master/`, `pdn/worker/` | Master con planificador, workers persistentes, latidos, fallos simulados, alta disponibilidad |
| Modo MPI | `pdn/mpi/` | Reparto estático, referencia secuencial, series de escalabilidad, Amdahl |
| Monitoreo | `pdn/monitoreo/` | Recursos (psutil) y energía (RAPL, NVML, powermetrics) |
| Interfaz | `pdn/dashboard/`, `pdn/cli.py`, `pdn/evidencias.py` | Dashboard web, línea de comandos, matriz de evidencias |
| Simulación | `pdn/sim/` | Clúster local con workers en procesos y retardos por MB |

## Una tarea, de punta a punta

1. El Master divide el espacio (el `.seq`, o la concatenación de parejas en la comparación) en unidades de 4 MiB.
2. Un worker pide trabajo (`PEDIR`). El planificador decide cuántas unidades le tocan: lo que procesaría en el tiempo objetivo según su velocidad medida (1 unidad al principio y menos en la fase final).
3. El Master envía `TAREA` con la carga: `inicio`, `fin`, `solape` y los límites de registro (o los segmentos de las parejas).
4. El worker procesa con su motor y devuelve `RESULTADO`: el parcial, el CRC de cada unidad, los tiempos y la energía de la tarea.
5. El Master verifica el CRC contra su tabla `.huellas`, combina el parcial, actualiza la velocidad del worker y responde con la siguiente tarea.
6. Si un worker deja de enviar latidos durante 5 s, o una tarea pasa de `max(3 × esperado, 10 s)`, la tarea vuelve a la fila y se registra la reasignación.
7. Al terminar: se comprueba la cobertura (cada unidad exactamente una vez), se compara con la referencia secuencial si existe y se guarda todo en `/cluster/resultados/`.

## Protocolo

Mensajes JSON sobre ZeroMQ (`pdn/comun/protocolo.py`). Canal principal, petición-respuesta: `REGISTRO`, `CALIBRACION`, `PEDIR`, `LISTO`, `RESULTADO` y `ERROR` (worker → Master); `ACEPTADO`, `PREPARAR`, `TAREA`, `ESPERAR`, `FIN` y `DETENER` (Master → worker). Canal de latidos: `LATIDO` (worker → Master) y las órdenes asíncronas `SIMULAR_FALLO` y `DETENER` (Master → worker). Replicación: `INSTANTANEA` y `ACUSE`, entre el principal y el respaldo.
