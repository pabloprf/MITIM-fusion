"""
Run control of gyrokinetic simulations (CGYRO, and GX where shared): what wraps, watches, stops,
accepts, rescues, resubmits and schedules a running radius, and the names of the files it leaves.

    markers       every tag/file name and token the run-control machinery reads or writes
    shell         template loader for the bash shipped to the compute nodes
    watchdog      the supervisor around one radial launch and the markers/exit verdict around it
    cost_guard    stop of a radius that is slow and far above its target
    completion    "did this radius run to completion?"
    status        per-radius state of a running submission (remote probe)
    rescue        stalled radii (scancel + resubmit) and interrupted runs (continue in place)
    submission    re-attach to, poll and fetch a submission; the per-poll callback
    extra_points  extra cases on freed slots, bash (in-allocation scheduler) and array modes
    cgyro_client  GKRunControl, the run-control half of the CGYRO class
    live          live state of the radii of a running evaluation, stop requests
"""
