#!/usr/bin/env bash
# Prepara la Mac (worker NPU): Homebrew, Python 3.12, entorno, coremltools y la
# regla de sudoers para powermetrics. Uso: bash scripts/setup_mac.sh
set -euo pipefail
# shellcheck source=scripts/comun.sh
source "$(dirname "$0")/comun.sh"

[[ "$(uname -s)" == "Darwin" ]] || error "este script es para macOS"
[[ "$(uname -m)" == "arm64" ]] || aviso "no es Apple Silicon: no habra Neural Engine"

info "Homebrew y Python"
command -v brew >/dev/null 2>&1 || error "instale Homebrew: https://brew.sh"
ok "Homebrew $(brew --version | head -1)"
# coremltools publica ruedas para versiones concretas de Python: se usa 3.12
if ! brew list python@3.12 >/dev/null 2>&1; then
    brew install python@3.12
fi
PY312="$(brew --prefix python@3.12)/bin/python3.12"
ok "Python $("$PY312" --version)"

ENTORNO="$HOME/pdn-env"
info "Entorno $ENTORNO"
[[ -x "$ENTORNO/bin/python" ]] || "$PY312" -m venv "$ENTORNO"
"$ENTORNO/bin/python" -m pip install -q --upgrade pip
"$ENTORNO/bin/python" -m pip install -q -r "$RAIZ/requirements/base.txt" -r "$RAIZ/requirements/mac.txt"
if "$ENTORNO/bin/python" -c "import coremltools; print(coremltools.__version__)"; then
    ok "coremltools importa"
else
    falla "coremltools no importa con esta version de Python"
fi

info "powermetrics sin contrasena (solo ese comando)"
REGLA="$(id -un) ALL=(root) NOPASSWD: /usr/bin/powermetrics"
if sudo -n /usr/bin/powermetrics -n 1 -i 100 --samplers cpu_power >/dev/null 2>&1; then
    ok "sudo -n powermetrics funciona"
else
    aviso "agregue esta linea con 'sudo visudo -f /etc/sudoers.d/pdn-powermetrics':"
    echo "      $REGLA"
    aviso "y vuelva a correr este script para verificarla"
fi

info "SSH (Compartir > Inicio de sesion remoto)"
if sudo -n systemsetup -getremotelogin 2>/dev/null | grep -qi "on"; then ok "inicio de sesion remoto activo"; else aviso "active 'Inicio de sesion remoto' en Ajustes > General > Compartir"; fi
