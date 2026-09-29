_lb_dir="@{rhodir}"; _lb_budget=@{budget_s}; _lb_min=@{min_time}; _lb_discard=@{discard}
set -m 2>/dev/null
# a stop request applies to this launch only (a leftover one would end it at the first restart write)
rm -f "$_lb_dir/mitim_stop"
(
@{cgyro_cmd}
) & _lb_pid=$!
_lb_t0=$(date +%s)
_lb_mt() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null; }
_lb_tree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; _lb_tree "$c"; done; }
# TERM the launch and wait until its whole process tree is gone: mpirun/prterun puts every rank in
# its own process group, so the group kill alone returns while ranks still hold the node/GPUs
_lb_kill() {
    local pids="$_lb_pid $(_lb_tree $_lb_pid)" i
    kill -TERM -- -$_lb_pid 2>/dev/null || pkill -TERM -P $_lb_pid
    for i in $(seq 1 60); do kill -0 $pids 2>/dev/null || return 0; sleep 1; done
    kill -KILL -- -$_lb_pid 2>/dev/null; kill -KILL $pids 2>/dev/null
}
while kill -0 $_lb_pid 2>/dev/null; do
    for _lb_i in $(seq 1 20); do kill -0 $_lb_pid 2>/dev/null || break; sleep 1; done
    kill -0 $_lb_pid 2>/dev/null || break
    # stop request: the wall budget (if any) is spent, or mitim_stop was dropped (scheduler, mitim_kill_cgyro)
    _lb_stop=0; [ -f "$_lb_dir/mitim_stop" ] && _lb_stop=1
    if (( _lb_stop == 0 )); then (( _lb_budget > 0 )) || continue; (( $(date +%s) - _lb_t0 < _lb_budget )) && continue; fi
    _lb_t=$(tail -n1 "$_lb_dir/out.cgyro.time" 2>/dev/null | awk '{print $1+0}')
    if ! awk -v t="${_lb_t:-0}" -v m="$_lb_min" 'BEGIN{exit !(t>=m)}'; then
        (( _lb_stop == 0 )) && continue
        # a main radius is never discarded (the evaluation needs it): stopped below min_time it is accepted like any stop
        if (( _lb_discard == 1 )); then
        echo "DISCARD t=${_lb_t:-0} min_time=$_lb_min elapsed=$(( $(date +%s) - _lb_t0 ))s" > "$_lb_dir/mitim_discard.tag"
        _lb_kill
        break
        fi
    fi
    # past min_time: stop right after the next restart write (out.cgyro.tag is rewritten with
    # every bin.cgyro.restart) so the blob a later iteration warm-starts from is whole
    _lb_m0=$(_lb_mt "$_lb_dir/out.cgyro.tag")
    while kill -0 $_lb_pid 2>/dev/null && [ "$(_lb_mt "$_lb_dir/out.cgyro.tag")" = "$_lb_m0" ]; do sleep 5; done
    sleep 5
    _lb_t=$(tail -n1 "$_lb_dir/out.cgyro.time" 2>/dev/null | awk '{print $1+0}')
    echo "$([ $_lb_stop = 1 ] && echo STOP || echo BUDGET) t=$_lb_t elapsed=$(( $(date +%s) - _lb_t0 ))s budget=${_lb_budget}s min_time=$_lb_min" > "$_lb_dir/mitim_budget.tag"
    _lb_kill
    break
done
wait $_lb_pid 2>/dev/null
