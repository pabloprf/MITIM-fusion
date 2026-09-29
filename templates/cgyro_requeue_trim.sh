# A SLURM requeue (preemption, node failure) reruns this same script in the same folder: CGYRO resumes
# from out.cgyro.tag and would add the FULL MAX_TIME again. The first launch of a submission records the
# time the run must reach; a relaunch under the same job trims MAX_TIME to what is left, rounded up to
# whole restart periods so RESTART_STEP keeps firing (overshoot below one period).
_rq_dir="@{run_dir}"; _rq_in="$_rq_dir/input.cgyro"; _rq_job="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}_${SLURM_ARRAY_TASK_ID:-0}"
_rq_key() { awk -F= -v k="$1" '{g=$1; gsub(/[ \t]/,"",g)} g==k {v=$2; gsub(/[ \t]/,"",v); print v+0; exit}' "$_rq_in" 2>/dev/null; }
_rq_mt=$(_rq_key MAX_TIME)
if [ -n "$_rq_mt" ]; then
    if [ "$(awk '{print $1}' "$_rq_dir/.mitim_t_end" 2>/dev/null)" = "$_rq_job" ]; then
        _rq_end=$(awk '{print $2}' "$_rq_dir/.mitim_t_end")
        _rq_new=$(awk -v te="$_rq_end" -v t0="${_t0:-0}" -v dt="$(_rq_key DELTA_T)" -v ps="$(_rq_key PRINT_STEP)" -v rs="$(_rq_key RESTART_STEP)" \
            'BEGIN{r=te-t0; p=rs*ps*dt; if (p>0) {n=int(r/p); if (n*p < r-1e-9*p) n++; if (n<1) n=1; r=n*p} else if (r<=0) r=ps*dt; printf "%.10g\n", r}')
        if [ "$_rq_new" != "$_rq_mt" ]; then
            sed -i.mitim_bak "s/^\([ \t]*MAX_TIME[ \t]*=[ \t]*\).*/\1$_rq_new/" "$_rq_in" && rm -f "$_rq_in.mitim_bak"
            echo "MITIM: requeued at t=${_t0:-0}, run ends at t=$_rq_end: MAX_TIME $_rq_mt -> $_rq_new" >&2
        fi
    else
        awk -v j="$_rq_job" -v a="${_t0:-0}" -v b="$_rq_mt" 'BEGIN{printf "%s %.10g\n", j, a+b}' > "$_rq_dir/.mitim_t_end"
    fi
fi
