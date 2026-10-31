#!/bin/sh
''''true
# ---------------- shell part: find / prepare Python, then run this same file with it ----------------
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"; HERE="$(dirname "$SELF")"
NEED="epyt numpy pandas openpyxl matplotlib plotly"
if command -v nvidia-smi >/dev/null 2>&1; then NEED="$NEED torch"; fi        # NVIDIA GPU -> PyTorch (GPU training)
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then echo "python3 is required (e.g. sudo apt install python3)"; exit 1; fi
CHECK="import epyt, numpy, pandas, openpyxl, matplotlib"
case "$NEED" in *torch*) CHECK="$CHECK, torch";; esac
if [ -x "$HERE/.leak_env/bin/python" ] && "$HERE/.leak_env/bin/python" -c "$CHECK" >/dev/null 2>&1; then
    PY="$HERE/.leak_env/bin/python"
elif ! "$PY" -c "$CHECK" >/dev/null 2>&1; then
    echo "First start: installing the required packages into $HERE/.leak_env (internet needed once) ..."
    case "$NEED" in *torch*) echo "  (NVIDIA GPU found -> PyTorch with GPU support is installed too, about 2-3 GB)";; esac
    [ -x "$HERE/.leak_env/bin/python" ] || "$PY" -m venv "$HERE/.leak_env" || { echo "Could not create the environment (install python3-venv), or run:  $PY -m pip install --user $NEED"; exit 1; }
    "$HERE/.leak_env/bin/python" -m pip install --quiet --disable-pip-version-check --no-warn-script-location --upgrade pip >/dev/null 2>&1
    for P in $NEED; do echo "  installing $P ..."; "$HERE/.leak_env/bin/python" -m pip install --quiet --disable-pip-version-check --no-warn-script-location "$P" || { echo "Package installation failed ($P)"; exit 1; }; done
    echo "  all packages installed."
    PY="$HERE/.leak_env/bin/python"
fi
exec "$PY" "$SELF" "$@"
'''
