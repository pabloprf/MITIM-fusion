"""
`mitim_simulation._machine_limits` caps cores_per_node of a "local" machine to the SLURM
allocation the driver sits in, only when the code really runs in that allocation (bash).

Checks:
  1. A slurm block without a partition (Perlmutter: qos/account/constraint) is a SLURM launch:
     no cap, even inside a 16-CPU driver job.
  2. A slurm block with a partition: no cap (unchanged behavior).
  3. launch_slurm False (bash): capped to SLURM_NTASKS x SLURM_CPUS_PER_TASK.
  4. An empty slurm block: bash, capped.

Run: python tests/dev_tests/test_machine_limits_slurm.py
"""

import os
import types
from unittest import mock

from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools import FARMINGtools

DRIVER_ENV = {"SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "16"}


def _limits(slurm_block, launch_slurm):
    machine = {"machine": "local", "cores_per_node": 128, "gpus_per_node": 0, "slurm": slurm_block}
    sim = types.SimpleNamespace(simulation_job=types.SimpleNamespace(machineSettings=machine))
    settings = types.SimpleNamespace(code="cgyro", launch_slurm=launch_slurm)
    with mock.patch.object(FARMINGtools.mitim_job, "grab_machine_settings", staticmethod(lambda code: dict(machine))), \
            mock.patch.dict(os.environ, DRIVER_ENV):
        return SIMtools.mitim_simulation._machine_limits(sim, settings)["cores_per_node"]


def test_perlmutter_block_without_partition_is_not_capped():
    block = {"account": "m4504", "qos": "regular", "constraint": "cpu"}
    assert _limits(block, launch_slurm=True) == 128


def test_block_with_partition_is_not_capped():
    assert _limits({"partition": "sched_mit_psfc_r8"}, launch_slurm=True) == 128


def test_bash_launch_is_capped_to_driver_allocation():
    block = {"account": "m4504", "qos": "regular", "constraint": "cpu"}
    assert _limits(block, launch_slurm=False) == 16


def test_empty_slurm_block_is_capped():
    assert _limits({}, launch_slurm=True) == 16


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
