# Runs in the GX run folder right before `gx`. A SLURM requeue (preemption, node failure) reruns this same script in
# the same folder: GX then resumes by itself from the gxplasma.restart.nc it wrote there every nsave steps
# (restart_if_exists in input.gx.controls) and stops at the same absolute t_max, so only the remaining time runs.
# Done here because GX does not:
#   - a restart file cut short by the preemption would stop GX when it opens it: drop it (the run starts over);
#   - append the resumed segment to gxplasma.out.nc instead of overwriting it (append_on_restart must stay false on a
#     first launch, when there is no gxplasma.out.nc to append to). Rows between the checkpoint and the preemption
#     are written twice; GXoutput keeps the later ones.
if [ -f gxplasma.restart.nc ] && command -v ncdump > /dev/null 2>&1 && ! ncdump -h gxplasma.restart.nc > /dev/null 2>&1; then
    echo "MITIM: gxplasma.restart.nc is unreadable (cut by the preemption?); removed, GX starts over" >&2
    rm -f gxplasma.restart.nc
fi
if [ -f gxplasma.restart.nc ] && [ -f gxplasma.out.nc ]; then
    sed -i 's/^\([ \t]*append_on_restart[ \t]*=\).*/\1 true/' gxplasma.in
    echo "MITIM: requeued run, resuming from gxplasma.restart.nc and appending to gxplasma.out.nc" >&2
fi
