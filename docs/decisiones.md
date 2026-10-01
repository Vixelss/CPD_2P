# Decisiones de diseño

Cada decisión no obvia, con su motivo y la alternativa descartada. Alimenta la sección del informe que contrasta la solución propia con la de una IA.

| # | Etapa | Decisión | Motivo | Alternativa descartada |
|---|---|---|---|---|
| 1 | E0 | Los cuatro proyectos del Parcial 1 se mueven a `referencias/` con `git mv` y se deja una sola copia del enunciado como `referencias/enunciado.pdf`. | Conservar el historial y no versionar cuatro copias de un PDF de 12 MB. | Borrar las carpetas y copiar solo el código útil (se perdería el historial). |
| 2 | E0 | Se normalizan los finales de línea a LF con `.gitattributes`. | Varios archivos del P1.4 venían con CRLF de Windows. | Dejar CRLF (diffs ruidosos en Linux). |
