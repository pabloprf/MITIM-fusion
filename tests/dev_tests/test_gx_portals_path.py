"""
PORTALS-GX path: the GX output carries every flux the shared gyrokinetic collector reads (and the electron
turbulent exchange), the input writer places the physics keys GX reads (instead of a batch-fatal "not written"
prompt), each radius gets its own array element on one node, and the warm start / requeue resume pieces
(restart file staging, t_max on a warm start, restart retrieval, the requeue bash, rewound output rows).
"""
import subprocess

import numpy as np
import netCDF4

from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import SLURMtools
from mitim_tools.simulation_tools.physics import GXtools
from mitim_modules.powertorch.physics_models.transport_cgyro import _FLUX_SPEC, _EXCHANGE_SPEC
from mitim_modules.powertorch.physics_models.utils.cgyro_restart import RestartChain


def _write_gx_output(folder, nt=40, nsp=3, nky=4, time=None, heating=False):
    '''
    Minimal gxplasma.out.nc: species in MITIM order (ions, electrons last), constant fluxes Q_s = s, G_s = 0.1*s,
    and with `heating` a TurbulentHeating_st that sums to zero over species (electron +0.5).
    '''
    time = np.linspace(0.0, 100.0, nt) if time is None else np.asarray(time)
    nt = len(time)
    ds = netCDF4.Dataset(folder / "gxplasma.out.nc", "w")
    for name, n in (("time", nt), ("s", nsp), ("ky", nky), ("kx", 1), ("ri", 2), ("theta", 8)):
        ds.createDimension(name, n)
    grids = ds.createGroup("Grids")
    grids.createVariable("time", "f8", ("time",))[:] = time
    grids.createVariable("theta", "f8", ("theta",))[:] = np.linspace(-np.pi, np.pi, 8)
    grids.createVariable("ky", "f8", ("ky",))[:] = 0.08 * np.arange(nky)
    diags = ds.createGroup("Diagnostics")
    diags.createVariable("omega_kxkyt", "f8", ("time", "ky", "kx", "ri"))[:] = 0.0
    Q = np.tile(np.arange(1, nsp + 1, dtype=float), (nt, 1))
    G = 0.1 * Q
    diags.createVariable("HeatFlux_st", "f8", ("time", "s"))[:] = Q
    diags.createVariable("ParticleFlux_st", "f8", ("time", "s"))[:] = G
    diags.createVariable("HeatFlux_kyst", "f8", ("time", "s", "ky"))[:] = np.repeat(Q[:, :, None] / nky, nky, axis=2)
    diags.createVariable("ParticleFlux_kyst", "f8", ("time", "s", "ky"))[:] = np.repeat(G[:, :, None] / nky, nky, axis=2)
    if heating:
        H = np.tile(np.append(-0.5 / (nsp - 1) * np.ones(nsp - 1), 0.5), (nt, 1))
        diags.createVariable("TurbulentHeating_st", "f8", ("time", "s"))[:] = H
    ds.createGroup("Inputs").createGroup("Controls").createVariable("nonlinear_mode", "i4").assignValue(1)
    ds.close()


def test_gx_output_carries_every_flux_the_collector_reads(tmp_path):
    _write_gx_output(tmp_path)
    out = GXtools.GXoutput(tmp_path, tmin=-0.5)

    for _, mean_attr, std_attr, _ in _FLUX_SPEC:
        assert hasattr(out, mean_attr) and hasattr(out, std_attr), mean_attr
    assert np.isclose(out.Qe_mean, 3.0)
    assert np.isclose(out.Qi_mean, 3.0)
    assert np.allclose(out.Gi_all_mean, [0.1, 0.2])   # per ion, input order: GZ indexes it by impurity position
    assert out.Mt_mean == 0.0


def test_gx_writer_places_physics_keys_in_their_blocks(tmp_path):
    inp = GXtools.GXinput()
    inp.controls.update({"x0": 15.0, "jtwist": 4, "fapar": 1.0, "fbpar": 1.0, "g_exb": 0.0, "cfl": 0.8, "ei_colls": True})
    inp.write_state(tmp_path / "gxplasma.in")

    block, where = None, {}
    for line in (tmp_path / "gxplasma.in").read_text().splitlines():
        line = line.strip()
        if line.startswith("["):
            block = line
        elif "=" in line and not line.startswith("#"):
            where[line.split("=")[0].strip()] = block
    assert where["x0"] == where["jtwist"] == "[Domain]"
    assert where["fapar"] == where["fbpar"] == where["g_exb"] == where["ei_colls"] == "[Physics]"
    assert where["cfl"] == "[Time]"


def test_gx_radii_get_one_array_element_each_on_one_node():
    machine = {"cores_per_node": 64, "gpus_per_node": 4, "slurm": {"partition": "ou_psfc_preemptable"}}
    # 2 radii x 1 GPU would fit one node (the slurm_standard heuristic), where both runs would sit on GPU 0
    r = SLURMtools.resolve("gx", allocation={"resources_per_call": 1, "minutes": 60, "max_concurrent_calls": 1},
                           n_rhos=2, machine_settings=machine, array_list=["0", "1"], verbose=False)
    assert r.submission_type == "slurm_array"
    assert r.sbatch["ntasks"] == 1 and r.sbatch["nodes"] == 1 and r.sbatch["gpus-per-task"] == 1
    assert r.sbatch["array"] == "0,1" and r.sbatch["array_limit"] == 1

    r = SLURMtools.resolve("gx", allocation={"resources_per_call": 4, "minutes": 60},
                           n_rhos=5, machine_settings=machine, array_list=[str(i) for i in range(5)], verbose=False)
    assert r.sbatch["ntasks"] == 4 and r.sbatch["nodes"] == 1
    assert "exclusive" not in r.sbatch


def test_gx_output_reads_the_electron_exchange_and_drops_rewound_rows(tmp_path):
    # a requeued run appended from its checkpoint at t=60 after reaching t=70: rows 60..70 of the first segment go
    time = np.concatenate([np.arange(0.0, 71.0, 10.0), np.arange(60.0, 101.0, 10.0)])
    _write_gx_output(tmp_path, time=time, heating=True)
    out = GXtools.GXoutput(tmp_path, tmin=-0.5)

    assert np.array_equal(out.t, np.arange(0.0, 101.0, 10.0)), out.t
    assert len(out.Qe) == len(out.t) and out.f.shape[-1] == len(out.t)
    _, mean_attr, std_attr, _ = _EXCHANGE_SPEC
    assert np.isclose(getattr(out, mean_attr), 0.5) and hasattr(out, std_attr)
    assert np.isclose(out.Si_mean, -0.5)   # exchange is conserved over species
    assert "Se_mean" in out.harvest_outputs()


def test_gx_output_without_heating_has_no_exchange(tmp_path):
    _write_gx_output(tmp_path)
    assert not hasattr(GXtools.GXoutput(tmp_path, tmin=-0.5), "Se_mean")   # collector passes QieGB_turb = 0


def _write_restart(path, t):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createVariable("time", "f8").assignValue(t)


def test_gx_warm_start_t_max_is_added_to_the_saved_time(tmp_path):
    gx = GXtools.GX(rhos=[0.3, 0.5])
    _write_restart(tmp_path / "gxplasma.restart.nc_0.3000", 123.0)
    files = {0.3: [(tmp_path / "gxplasma.restart.nc_0.3000", "gxplasma.restart.nc")]}

    assert gx._warm_start_t_max("Nonlinear_reduced3_analogue", {"t_max": 50.0}, files)["t_max"] == [173.0, 50.0]
    # from the preset when extraOptions does not set it (reduced2/3 analogues: 1200)
    assert gx._warm_start_t_max("Nonlinear_reduced3_analogue", {}, files)["t_max"] == [1323.0, 1200.0]
    # nothing staged under the GX name: untouched
    assert gx._warm_start_t_max("Nonlinear_reduced3_analogue", {"t_max": 50.0}, {0.3: [tmp_path / "other"]}) == {"t_max": 50.0}


def test_gx_restart_file_is_kept_only_when_a_chain_reads_it(tmp_path):
    gx = GXtools.GX(rhos=[0.3])
    for _ in range(2):   # idempotent across run() calls
        gx._restart_retrieval({})
    assert "gxplasma.restart.nc" in gx.output_files_simulation["complete"]
    assert "gxplasma.restart.nc" not in gx.output_files_simulation["optional"]

    gx.keep_warm_start_file = True
    gx._restart_retrieval({})
    assert gx.output_files_simulation["optional"] == ["gxplasma.restart.nc"]
    assert "gxplasma.restart.nc" not in gx.output_files_simulation["complete"]

    gx._restart_retrieval({"save_for_restart": False})
    assert "gxplasma.restart.nc" not in gx.output_files_simulation["complete"] + gx.output_files_simulation["optional"]


def test_restart_chain_stages_the_gx_restart_file(tmp_path):
    rhos = [0.3, 0.5]
    src = tmp_path / "Execution" / "Evaluation.0" / "transport_simulation_folder" / "base_gx"
    src.mkdir(parents=True)
    for rho in rhos:
        _write_restart(src / f"gxplasma.restart.nc_{rho:.4f}", 10.0)
    folder = tmp_path / "Execution" / "Evaluation.1" / "transport_simulation_folder"
    folder.mkdir(parents=True)

    chain = RestartChain({"restart_from_cases": "first"}, 1, folder, rhos, base_subfolder="base_gx",
                         restart_file=GXtools.GX._warm_start_file, label="GX")
    assert chain.active
    plan = chain.resolve(None, None)
    assert plan.sources == {"0.3000": 0, "0.5000": 0}
    assert all(dst == "gxplasma.restart.nc" for entries in plan.files_per_rho.values() for _, dst in entries)
    assert (folder / "base_gx" / "restart_sources.json").is_file()
    assert not RestartChain({}, 1, folder, rhos).active


def _bash(folder, script):
    return subprocess.run(["bash", "-c", script], cwd=folder, capture_output=True, text=True)


def test_requeue_bash_appends_only_when_resuming_its_own_run(tmp_path):
    body = (__mitimroot__ / "templates" / "gx_requeue_resume.sh").read_text()
    call = GXtools.GX(rhos=[0.3]).run_specifications["code_call"]("rho_0.3000", n=1, p="/scratch")
    assert body in call and ">> gxplasma.mitim.log" in call
    # multi-rank GX writes parallel HDF5; without this Open MPI locks every write on NFS and the run stalls
    assert call.index("export OMPI_MCA_fs_ufs_lock_algorithm=1") < call.index("gx -n")

    (tmp_path / "gxplasma.in").write_text("[Restart]\n append_on_restart       = false\n")
    _bash(tmp_path, body)                                  # first launch, cold
    assert "= false" in (tmp_path / "gxplasma.in").read_text()

    _write_restart(tmp_path / "gxplasma.restart.nc", 5.0)  # first launch, warm (staged parent, no output yet)
    _bash(tmp_path, body)
    assert "= false" in (tmp_path / "gxplasma.in").read_text()

    (tmp_path / "gxplasma.out.nc").write_text("x")         # requeue: own checkpoint and output
    r = _bash(tmp_path, body)
    assert "append_on_restart       = true" in (tmp_path / "gxplasma.in").read_text(), r.stderr
    assert (tmp_path / "gxplasma.restart.nc").is_file()

    if _bash(tmp_path, "command -v ncdump").returncode == 0:
        (tmp_path / "gxplasma.restart.nc").write_text("cut by the preemption")
        _bash(tmp_path, body)
        assert not (tmp_path / "gxplasma.restart.nc").exists()


def test_big_nc_expected_only_when_gx_writes_it():
    gx = GXtools.GX(rhos=[0.3])
    gx._big_retrieval("Nonlinear_reduced2_analogue", {})           # fields/moments off in the analogue presets
    gx._big_retrieval("Nonlinear_reduced2_analogue", {})
    assert "gxplasma.big.nc" not in gx.output_files_simulation["complete"]
    gx._big_retrieval("Nonlinear_reduced2_analogue", {"fields": True})
    gx._big_retrieval("Nonlinear_reduced2_analogue", {"fields": True})
    assert gx.output_files_simulation["complete"].count("gxplasma.big.nc") == 1


def test_nx_with_a_large_prime_nkx_is_refused():
    gx = GXtools.GX(rhos=[0.3])
    gx._check_nkx("Nonlinear_reduced2_analogue", {})                  # nx 384 -> nkx 255 = 3*5*17
    gx._check_nkx("Nonlinear_reduced3_analogue", {})                  # nx 294 -> nkx 195 = 3*5*13
    try:
        gx._check_nkx("Nonlinear_reduced3_analogue", {"nx": 288})    # nkx 191, prime: GX aborts at jtwist = 1
    except Exception as e:
        assert "191" in str(e)
    else:
        raise AssertionError("nx 288 accepted")
