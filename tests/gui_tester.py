"""Drive a viser script's GUI headlessly: wiggle every input, click a sphere, press every button.

Any exception raised by a callback fails the run. Used by docker_smoke.sh:

    python3 gui_tester.py <script.py> [script args...]
"""
import runpy
import sys
import time

import viser

handles = []  # (kind, handle) in creation order
spheres = []

for kind in ("button", "slider", "checkbox", "dropdown"):
    original = getattr(viser.GuiApi, f"add_{kind}")

    def record(self, *args, _original = original, _kind = kind, **kwargs):
        handle = _original(self, *args, **kwargs)
        handles.append((_kind, handle))
        return handle

    setattr(viser.GuiApi, f"add_{kind}", record)

add_icosphere = viser.SceneApi.add_icosphere


def record_sphere(self, *args, **kwargs):
    handle = add_icosphere(self, *args, **kwargs)
    spheres.append(handle)
    return handle


viser.SceneApi.add_icosphere = record_sphere


class SetupDone(Exception):
    pass


def stop(*_):
    raise SetupDone


time.sleep = stop  # every script's main loop sleeps; the first sleep means the GUI is built


def fire(handle):
    for callback in list(handle._impl.update_cb):
        callback(viser.GuiEvent(client_id = None, client = None, target = handle))


def label(handle):
    return handle.label


script, *argv = sys.argv[1:]
sys.argv = [script, *argv, "--port", "18181"]
try:
    runpy.run_path(script, run_name = "__main__")
except SetupDone:
    pass

# Wiggle every input across its range, then restore it (setting .value fires its callbacks).
for kind, handle in handles:
    if handle.disabled:
        continue

    value = handle.value
    if kind == "slider":
        trials = [handle.min, handle.max]
    elif kind == "checkbox":
        trials = [not value]
    elif kind == "dropdown":
        trials = list(handle.options)
    else:
        continue

    for trial in trials + [value]:
        handle.value = trial

    print(f"  ok  {kind}: {label(handle)}")



def click_sphere():
    """Click a live sphere if there is one (the editor selects on click); ignored while one is selected."""
    clickable = [h for h in spheres if h._impl.click_cb and not h._impl.removed]
    if clickable:
        clickable[0]._impl.click_cb[0].callback(None)


# Press one button at a time in creation order, re-clicking a sphere before each, until nothing new
# is enabled -- so buttons that need a selection (Delete) or a prior step (Spherize -> Export) get pressed.
pressed = set()
while True:
    click_sphere()
    ready = [h for k, h in handles if k == "button" and not h.disabled and id(h) not in pressed]
    if not ready:
        break

    pressed.add(id(ready[0]))
    fire(ready[0])
    print(f"  ok  button: {label(ready[0])}")

# Writers ran before later edits (e.g. Save before Delete); press them again so the files hold the final state.
for kind, handle in handles:
    if kind == "button" and id(handle) in pressed and label(handle).startswith(("Save", "Export")):
        fire(handle)

# Buttons that never became enabled (e.g. Delete without a selection) are fine; list them for visibility.
for kind, handle in handles:
    if kind == "button" and id(handle) not in pressed:
        print(f"  --  button never enabled: {label(handle)}")

print("GUI OK")
