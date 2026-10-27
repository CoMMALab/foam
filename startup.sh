#!/bin/bash
set -e

# Dev mode: a source checkout mounted over /foam has no compiled binaries yet
if [ ! -x /foam/foam/external/makeTreeMedial ]; then
    (cd /foam && cmake -Bbuild -GNinja . && cmake --build build/ && uv pip install -e . --no-deps) >&2
fi

# `<script> [args]` -> python /foam/scripts/<script>.py [args]
if [ -n "$1" ] && [ -f "/foam/scripts/${1%.py}.py" ]; then
    script="/foam/scripts/${1%.py}.py"
    shift
    exec python3 "$script" "$@"
fi

exec "$@"
