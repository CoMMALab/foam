#!/bin/bash
# Smoke-test every foam command in the Docker image (see DOCKER_QUICKSTART.md).
# Works on a temp copy of assets/panda, so nothing is written to the repo; the copy is deleted on exit.
#
#   bash tests/docker_smoke.sh            # uses existing `foam` image
#   BUILD=1 bash tests/docker_smoke.sh    # rebuild first
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-foam}"
PORT="${PORT:-18080}"
DATA="$(mktemp -d)"
cp -r "$REPO/assets/panda" "$DATA/"
trap 'docker rm -f foam-smoke >/dev/null 2>&1; rm -rf "$DATA"' EXIT

[ "${BUILD:-0}" = 1 ] && docker build -t "$IMAGE" "$REPO"

RUN=(docker run --rm -u "$(id -u):$(id -g)" -v "$DATA:/data" "$IMAGE")
pass=0; fail=0; failed=()

report() {  # name, ok(0/1), log
    if [ "$2" = 0 ]; then
        echo "PASS  $1"; pass=$((pass + 1))
    else
        echo "FAIL  $1"; echo "$3" | tail -5 | sed 's/^/      /'; fail=$((fail + 1)); failed+=("$1")
    fi
}

# name, expected output file (relative to /data), command...
check_cli() {
    local name=$1 out=$2; shift 2
    local log; log=$("${RUN[@]}" "$@" 2>&1)
    [ $? = 0 ] && [ -s "$DATA/$out" ]
    report "$name" $? "$log"
}

# name, command...: serves http 200 within 60s
check_web() {
    local name=$1; shift
    docker run --rm -d --name foam-smoke -p "$PORT:8080" -u "$(id -u):$(id -g)" -v "$DATA:/data" "$IMAGE" "$@" >/dev/null
    local ok=1
    for _ in $(seq 60); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' "localhost:$PORT")" = 200 ] && { ok=0; break; }
        docker inspect foam-smoke >/dev/null 2>&1 || break  # crashed
        sleep 1
    done
    local log; log=$(docker logs foam-smoke 2>&1)
    docker rm -f foam-smoke >/dev/null 2>&1
    report "$name" $ok "$log"
}

MESH=panda/meshes/collision/link4.obj

check_cli "generate_spheres"                link4-spheres.json         generate_spheres $MESH
check_cli "generate_spheres (spawn, d2 b12)" link4-spawn.json          generate_spheres $MESH --method spawn --depth 2 --branch 12 --output link4-spawn.json
check_cli "generate_sphere_urdf"            panda/panda_spherized.urdf generate_sphere_urdf --filename panda/panda.urdf --output panda/panda_spherized.urdf
check_cli "generate_sphere_urdf (link scale)" panda/panda_hand2x.urdf  generate_sphere_urdf --filename panda/panda.urdf --output panda/panda_hand2x.urdf --panda_hand 2.0
check_cli "generate_srdf"                   panda/panda_spherized.srdf generate_srdf panda/panda_spherized.urdf --n_samples 2000
check_cli "generate_srdf (margin)"          panda/margin.srdf          generate_srdf panda/panda_spherized.urdf --contact_margin 0.01 --n_samples 500 --output panda/margin.srdf

check_web "visualize_spheres"         visualize_spheres $MESH link4-spheres.json
check_web "visualize_urdf"            visualize_urdf panda/panda_spherized.urdf
check_web "spherize_urdf_interactive" spherize_urdf_interactive panda/panda.urdf
check_web "edit_spheres"              edit_spheres panda/panda_spherized.urdf --output panda/panda_edited.urdf

# name, command...: drives the GUI headlessly (every input, a sphere click, every button), then checks
# the files the GUI writes. Mounts tests/ so editing gui_tester.py needs no rebuild.
check_gui() {
    local name=$1 out=$2; shift 2
    [ -n "$out" ] && rm -f "$DATA/$out"
    local log; log=$(docker run --rm -u "$(id -u):$(id -g)" -v "$DATA:/data" -v "$REPO/tests:/foam/tests:ro" \
        "$IMAGE" python3 /foam/tests/gui_tester.py "/foam/scripts/$1.py" "${@:2}" 2>&1)
    echo "$log" | grep -q "^GUI OK" && { [ -z "$out" ] || [ -s "$DATA/$out" ]; }
    report "$name" $? "$log"
}

check_gui "gui: visualize_spheres"         ""                    visualize_spheres $MESH link4-spheres.json
check_gui "gui: visualize_urdf"            ""                    visualize_urdf panda/panda_spherized.urdf
check_gui "gui: spherize_urdf_interactive" panda/panda_gui.urdf  spherize_urdf_interactive panda/panda.urdf --output panda/panda_gui.urdf
check_gui "gui: edit_spheres"              panda/panda_edited.urdf edit_spheres panda/panda_spherized.urdf --output panda/panda_edited.urdf

# The editor's Delete removed exactly one sphere from what Save wrote.
n_in=$(grep -c "<sphere" "$DATA/panda/panda_spherized.urdf"); n_out=$(grep -c "<sphere" "$DATA/panda/panda_edited.urdf" 2>/dev/null)
[ "$n_out" = $((n_in - 1)) ]; report "gui: edit_spheres deleted one sphere ($n_in -> ${n_out:-?})" $? ""

log=$("${RUN[@]}" python3 -c "import foam, foam.viz; print('ok')" 2>&1)
[ "$(echo "$log" | tail -1)" = ok ]; report "passthrough command" $? "$log"

log=$(docker run --rm -u "$(id -u):$(id -g)" -v "$REPO:/foam" -v "$DATA:/data" "$IMAGE" \
    generate_spheres $MESH --output dev.json 2>&1)
[ $? = 0 ] && [ -s "$DATA/dev.json" ]; report "dev mode (source mounted)" $? "$log"

echo
echo "$pass passed, $fail failed"
[ $fail = 0 ] || { printf '  - %s\n' "${failed[@]}"; exit 1; }
