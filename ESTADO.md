# ESTADO DEL PROYECTO

Traspaso entre sesiones de Claude Code. Debe bastar para retomar sin leer el historial. La especificación es `CONTEXTO.md`.

## Entorno de desarrollo usado en la nube

- Python 3.12 en `.venv/` (ignorado por Git): `python3.12 -m venv .venv && .venv/bin/pip install -r requirements/base.txt numba`.
- Pruebas: `.venv/bin/python -m pytest -q`.

## Etapas

| Etapa | Estado |
|---|---|
| E0 Reorganización y esqueleto | Hecha |
| E1 Datos y operaciones de referencia | Pendiente |
| E2 Motor CPU y núcleo SIMD | Pendiente |
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

## Pendiente de prueba en hardware

(Se completa en cada etapa.)

## Preguntas abiertas para el profesor

1. ¿El tiempo de distribuir el archivo a los nodos cuenta en el examen?
2. ¿Acepta Core ML con el Neural Engine de Apple como implementación de NPU, en lugar de OpenVINO o TensorRT?
3. Con el archivo de 10 GB del examen, ¿qué operación se pide exactamente?
