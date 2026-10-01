# CPD_2P — Clúster heterogéneo y distribuido para procesamiento de ADN (P2.3)

Sistema que reparte el procesamiento de genomas en formato FASTA entre varias laptops (CPU, GPU CUDA y NPU de Apple), con un Master que balancea la carga de forma adaptativa, tolera fallos y muestra todo en un dashboard web.

- Especificación completa: [`CONTEXTO.md`](CONTEXTO.md)
- Avance por etapa y traspaso: [`ESTADO.md`](ESTADO.md)
- Decisiones de diseño: [`docs/decisiones.md`](docs/decisiones.md)
- Proyectos del Parcial 1 (solo lectura): [`referencias/`](referencias/)

## Inicio rápido (desarrollo)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements/base.txt
.venv/bin/python -m pytest
```
