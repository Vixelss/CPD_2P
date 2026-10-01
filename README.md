# CPD_2P — Clúster heterogéneo y distribuido para procesamiento de ADN (P2.3)

Sistema que reparte el procesamiento de genomas FASTA (GRCh38, 3,1 GB) entre varias laptops: CPU (numpy y C con AVX2), GPU NVIDIA (Numba CUDA) y el Neural Engine de una Mac (Core ML con un modelo INT8). Un Master balancea la carga con raciones adaptativas, verifica la integridad por CRC, tolera la caída de workers y del propio Master (con un Master de respaldo) y muestra todo en un dashboard web. También tiene un modo MPI con reparto estático, la referencia secuencial y las series de escalabilidad con ajuste de Amdahl.

Operaciones: conteo y validación, búsqueda de patrones (con complemento inverso), comparación de dos cadenas (emparejada por secuencia o posicional) y zonas de interés (islas CpG).

| Documento | Para qué |
|---|---|
| [`MANUAL.md`](MANUAL.md) | Levantar y operar el clúster el día de la presentación |
| [`CONTEXTO.md`](CONTEXTO.md) | Especificación completa |
| [`ESTADO.md`](ESTADO.md) | Avance por etapa, pendientes de hardware real |
| [`docs/arquitectura.md`](docs/arquitectura.md) | Componentes y recorrido de una tarea |
| [`docs/decisiones.md`](docs/decisiones.md) | Decisiones de diseño y alternativas descartadas (para el contraste con la IA del informe) |
| [`referencias/`](referencias/) | Proyectos del Parcial 1 y enunciado (solo lectura) |

## Probarlo en una sola máquina

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements/base.txt numba
.venv/bin/python -m pytest -q                       # unas 190 pruebas, unos 2 minutos

# Clúster simulado: 4 workers con velocidades distintas y el dashboard en http://127.0.0.1:8000
.venv/bin/python -m pdn.cli sim --workers 4 --retardos 0,0.5,1,2 --mb 6 --dashboard
```

## Comandos principales

```bash
python -m pdn.cli preparar ARCHIVO.fna                      # .seq, .idx, .huellas
python -m pdn.cli master --dashboard --replicar-a IP_RESP   # Master (+ dashboard)
python -m pdn.cli respaldo --dashboard                      # Master de respaldo
python -m pdn.worker --master IP --respaldo IP_RESP --dispositivo cpu|gpu|npu
python -m pdn.cli correr --operacion conteo --archivo NOMBRE --url http://IP:8000
python -m pdn.cli correr --modo mpi --np 60 --hostfile hostfile --operacion conteo --archivo NOMBRE
python -m pdn.cli referencia --operacion conteo --archivo NOMBRE --repeticiones 3 --calentamiento
python -m pdn.mpi.escalabilidad --archivo NOMBRE --repeticiones 3 --calentamiento
```

Los scripts de despliegue están en [`scripts/`](scripts/) (ver `MANUAL.md`).
