"""
test_cgyro_cost_record.py
=========================
The per-radius wall cost that PORTALS-CGYRO writes into fluxes_turb.json
(additional_info.cost, transport_cgyro._cost_records): seconds per a/cs, wall seconds, hosts,
node-hours and node-hours per 900 a/cs, from out.cgyro.timing, out.cgyro.time and out.cgyro.hosts.

Run as:

    python tests/dev_tests/test_cgyro_cost_record.py
"""

import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

mitim_root = Path(__file__).resolve().parents[2]
if str(mitim_root / "src") not in sys.path:
    sys.path.insert(0, str(mitim_root / "src"))

from mitim_modules.powertorch.physics_models.transport_cgyro import _cost_records


def test_cost_record():
    '''Outputs 0.5 a/cs apart at 30 s each on 2 hosts: 60 s per a/cs, 30 node-h per 900 a/cs.'''
    d = Path(tempfile.mkdtemp())
    try:
        (d / "out.cgyro.hosts").write_text("rank=0 host=nid001\nrank=1 host=nid001\nrank=2 host=nid002\n")
        (d / "mitim_budget.tag").write_text("GUARD t=100 cost=60s_per_acs\n")
        out = SimpleNamespace(folder=d, t=0.5 * np.arange(1, 201), timing_total=np.full(200, 30.0),
                              timing_setup={"str_init": 20.0})
        rec = _cost_records([out])[0]
        assert rec["cost_s_per_acs"] == 60.0 and rec["n_nodes"] == 2.0, rec
        assert rec["wall_s"] == 6020.0 and abs(rec["node_h"] - 6020.0 * 2 / 3600) < 1e-12, rec
        assert abs(rec["node_h_per_900acs"] - 30.0) < 1e-12 and rec["t_last"] == 100.0, rec
        assert rec["guard_stop"] == 1.0 and rec["budget_stop"] == 1.0, rec

        # no out.cgyro.hosts: the node counts are unknown and written as null, the rest stays
        (d / "out.cgyro.hosts").unlink()
        rec = _cost_records([out])[0]
        assert rec["n_nodes"] is None and rec["node_h"] is None and rec["cost_s_per_acs"] == 60.0, rec

        # an output without a timing table (GX) gives no record
        assert _cost_records([SimpleNamespace(folder=d)]) is None
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS test_cost_record")


if __name__ == "__main__":
    test_cost_record()
    print("\nALL PASS")
