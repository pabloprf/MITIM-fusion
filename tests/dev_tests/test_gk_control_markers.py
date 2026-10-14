"""
test_gk_control_markers.py
==========================
The bash templates and the on-node guard script cannot import gk_control.markers, so they spell the
marker names out. Every name they use must be the one the Python side reads or writes:

  - each template carries the markers it is known to use, spelled as in the registry;
  - every `<dir>/mitim_*` or `<dir>/.mitim_*` file a template touches is a registry value.

Run as:

    python tests/dev_tests/test_gk_control_markers.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from mitim_tools.simulation_tools.gk_control import markers, shell

USES = {
    "cgyro_watchdog.sh": [markers.BUDGET_TAG, markers.DISCARD_TAG, markers.STOP_REQUEST, markers.GUARD_JSON,
                          markers.GUARD_SCRIPT, markers.GUARD_STATUS, f'"{markers.GUARD_TOKEN} ',
                          markers.CASH_KARP_ABORT, markers.CGYRO_COMPLETION[0]],
    "cgyro_probe.sh": [markers.BUDGET_TAG, markers.T0],
    "cgyro_requeue_trim.sh": [markers.T_END],
    "cgyro_guard.py": [markers.GUARD_JSON, markers.T_END],
}


def registry_values():
    return {v for k, v in vars(markers).items() if k.isupper() and isinstance(v, str)}


def test_templates_spell_the_registry():
    for name, used in USES.items():
        text = (shell.TEMPLATES / name).read_text()
        missing = [m for m in used if m not in text]
        assert not missing, f"{name} does not contain {missing}"
    print("PASS test_templates_spell_the_registry")


def test_no_unregistered_marker_files():
    known = registry_values()
    for name in list(USES) + ["gx_requeue_resume.sh"]:
        text = (shell.TEMPLATES / name).read_text()
        found = set(re.findall(r'/(\.?mitim_[A-Za-z0-9_.]*[A-Za-z0-9])', text))
        found |= set(re.findall(r'"(\.?mitim_[a-z0-9_]+\.[a-z]+)"', text))
        unknown = sorted(f for f in found if f not in known)
        assert not unknown, f"{name} uses marker files missing from gk_control.markers: {unknown}"
    print("PASS test_no_unregistered_marker_files")


if __name__ == "__main__":
    test_templates_spell_the_registry()
    test_no_unregistered_marker_files()
