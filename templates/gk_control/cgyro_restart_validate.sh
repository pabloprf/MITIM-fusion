# A filesystem outage during a restart write left an empty out.cgyro.tag and a bin.cgyro.restart with a zeroed header,
# and the next launch crashed on them. CGYRO writes a checkpoint as: blob to bin.cgyro.restart.part, header into it,
# bin.cgyro.restart -> bin.cgyro.restart.old, .part -> bin.cgyro.restart, THEN out.cgyro.tag (i_current, t_current).
# So bin.cgyro.restart.old is the previous checkpoint, RESTART_STEP outputs earlier, and a tag write that failed leaves
# the tag of .old as the newest one the watchdog saw (.mitim_tag_last; .mitim_tag_prev is the one before). Before
# anything reads the tag, check both and fall back:
#   a) bad (or missing) blob, valid .old: .old goes back in place (bad blob kept as bin.cgyro.restart.bad) with ITS tag,
#      the newest history entry older than the bad write's tag (any entry when that tag never landed), else rebuilt
#      RESTART_STEP outputs before the bad write from out.cgyro.time;
#   b) valid blob, bad tag: the tag of this blob never landed. With a valid .old and history, the pair .old +
#      .mitim_tag_last is exact (one restart period of progress lost); else the tag is rebuilt one RESTART_STEP after
#      .mitim_tag_last, or at the newest restart point out.cgyro.time reached, from that row of out.cgyro.time;
#   c) nothing to rebuild from: no tag, so CGYRO warm-starts at t=0 from a valid blob, or cold-starts without one.
# Valid blob: >= 1024 bytes (the header) opening with a CGYRO restart magic (v3 140974129, v2 140906808 / 140916753;
# int32, native = little endian on the nodes). Valid tag: line 1 a positive integer, line 2 a number; a rebuilt tag has
# 2 lines (CGYRO reads delta_t, line 3, only if present). A repaired tag restarts the history. Restart points are
# counted from the launch start (.mitim_t0 of the launch that wrote them, the nearest out.cgyro.time row). Valid files,
# and a launch with neither tag nor blob, are left alone. Every action is one "MITIM restart_validate:" line on stderr.
# _rv_apply=0 (interrupted-run probe) changes nothing and prints the time the launch would resume from (empty: cold start).
_rv_dir="@{run_dir}"; _rv_apply=@{apply}; _rv_t=0
_rv_tag="$_rv_dir/out.cgyro.tag"; _rv_rs="$_rv_dir/bin.cgyro.restart"; _rv_last="$_rv_dir/.mitim_tag_last"; _rv_prev="$_rv_dir/.mitim_tag_prev"
_rv_tag_ok() { awk 'NR==1 {i = ($1 ~ /^[0-9]+$/ && $1+0 > 0)} NR==2 {t = ($1 ~ /^[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?$/)} END {exit !(i && t)}' "$1" 2>/dev/null; }
_rv_blob_ok() { [ -f "$1" ] && [ "$(wc -c < "$1" | awk '{print $1+0}')" -ge 1024 ] && case "$(od -An -t d4 -N4 "$1" | awk '{print $1}')" in 140974129|140906808|140916753) ;; *) false ;; esac; }
_rv_i() { awk 'NR==1 {print $1+0}' "$1"; }
_rv_row() { awk -v n="$1" 'NF {k++} k==n {print $1; exit}' "$_rv_dir/out.cgyro.time" 2>/dev/null; }
_rv_log() { [ "$_rv_apply" = 1 ] && echo "MITIM restart_validate: $*" >&2; }
_rv_run() { [ "$_rv_apply" = 1 ] && "$@@"; }
# the tag, from a history file or rebuilt (output i, time t); the history restarts at it
_rv_set() {
    _rv_t=${2:-$(awk 'NR==2 {print $1}' "$1")}
    if [ "$_rv_apply" = 1 ]; then
        if [ -n "$2" ]; then printf '%12d\n  %s\n' "$1" "$2" > "$_rv_tag"; else cp "$1" "$_rv_tag"; fi
        cp "$_rv_tag" "$_rv_last"; rm -f "$_rv_prev"
    fi
}
_rv_unset() { _rv_t=$1; _rv_run rm -f "$_rv_tag" "$_rv_last" "$_rv_prev"; }
_rv_swap() { [ -f "$_rv_rs" ] && _rv_run mv -f "$_rv_rs" "$_rv_rs.bad"; _rv_run mv -f "$_rv_rs.old" "$_rv_rs"; }
# newest restart point of the launch that wrote out.cgyro.time: every RESTART_STEP outputs after its start row
_rv_newest() {
    awk -v t0="$(cat "$_rv_dir/.mitim_t0" 2>/dev/null)" -v rs="$_rv_step" 'NF {n++; d = $1 - t0; d = d < 0 ? -d : d; if (t0+0 > 0 && (n == 1 || d < b)) {b = d; i0 = n}}
        END {print (rs > 0 && n > 0) ? i0 + int((n - i0) / rs) * rs : 0}' "$_rv_dir/out.cgyro.time" 2>/dev/null
}
# rebuilt tag at output $1 with the time of that row, or (no such row) no tag and the warm start at t=0
_rv_rebuild() {
    _rv_r=$( [ "${1:-0}" -gt 0 ] 2>/dev/null && _rv_row "$1")
    if [ -n "$_rv_r" ]; then _rv_set "$1" "$_rv_r"; _rv_log "$2; tag rebuilt from out.cgyro.time: output $1, t=$_rv_r"
    else _rv_unset 0; _rv_log "$2; NO TAG could be rebuilt (output ${1:-?}): WARM START AT t=0 from bin.cgyro.restart, simulated time lost"; fi
}
if [ -f "$_rv_tag" ] || [ -f "$_rv_rs" ]; then
    _rv_step=$(awk -F= '{g=$1; gsub(/[ \t]/,"",g)} g=="RESTART_STEP" {v=$2; gsub(/[ \t]/,"",v); print int(v); exit}' "$_rv_dir/input.cgyro" 2>/dev/null)
    _rv_tv=none; [ -f "$_rv_tag" ] && { _rv_tag_ok "$_rv_tag" && _rv_tv=ok || _rv_tv=bad; }
    _rv_bv=bad; _rv_blob_ok "$_rv_rs" && _rv_bv=ok
    if [ "$_rv_bv" = ok ] && [ "$_rv_tv" = ok ]; then
        _rv_t=$(awk 'NR==2 {print $1}' "$_rv_tag")
    elif [ "$_rv_bv" = ok ] && [ "$_rv_tv" = none ]; then
        :
    elif [ "$_rv_bv" = bad ] && _rv_blob_ok "$_rv_rs.old"; then
        _rv_msg="bin.cgyro.restart $([ -f "$_rv_rs" ] && echo "has no valid header (kept as bin.cgyro.restart.bad)" || echo "missing"), restored bin.cgyro.restart.old"
        _rv_swap
        # the bad write's own tag landed: .old is older than it
        _rv_lim=$([ "$_rv_tv" = ok ] && _rv_i "$_rv_tag")
        _rv_h=""
        for _rv_f in "$_rv_last" "$_rv_prev"; do
            if _rv_tag_ok "$_rv_f" && { [ -z "$_rv_lim" ] || [ "$(_rv_i "$_rv_f")" -lt "$_rv_lim" ]; }; then _rv_h=$_rv_f; break; fi
        done
        if [ -n "$_rv_h" ]; then
            _rv_hi=$(_rv_i "$_rv_h"); _rv_set "$_rv_h"; _rv_log "$_rv_msg with its tag from ${_rv_h##*/}: output $_rv_hi, t=$_rv_t"
        else
            _rv_bad=${_rv_lim:-$(_rv_newest)}
            _rv_rebuild "$([ "${_rv_step:-0}" -gt 0 ] 2>/dev/null && echo $(( ${_rv_bad:-0} - _rv_step )))" "$_rv_msg (bad write at output ${_rv_bad:-?}, RESTART_STEP ${_rv_step:-?})"
        fi
    elif [ "$_rv_bv" = ok ]; then
        _rv_msg="out.cgyro.tag is not a valid tag"
        if _rv_blob_ok "$_rv_rs.old" && _rv_tag_ok "$_rv_last"; then
            _rv_hi=$(_rv_i "$_rv_last"); _rv_swap; _rv_set "$_rv_last"
            _rv_log "$_rv_msg; its blob's tag never landed: continuing from bin.cgyro.restart.old (newest blob kept as bin.cgyro.restart.bad) with .mitim_tag_last: output $_rv_hi, t=$_rv_t"
        elif [ "${_rv_step:-0}" -gt 0 ] 2>/dev/null && _rv_tag_ok "$_rv_last"; then
            _rv_rebuild "$(( $(_rv_i "$_rv_last") + _rv_step ))" "$_rv_msg (one RESTART_STEP $_rv_step after .mitim_tag_last)"
        elif [ "${_rv_step:-0}" -gt 0 ] 2>/dev/null; then
            _rv_rebuild "$(_rv_newest)" "$_rv_msg (newest restart point, RESTART_STEP $_rv_step)"
        else
            _rv_unset 0; _rv_log "$_rv_msg and nothing to rebuild it from: WARM START AT t=0 from bin.cgyro.restart, simulated time lost"
        fi
    else
        [ -f "$_rv_rs" ] && _rv_run mv -f "$_rv_rs" "$_rv_rs.bad"
        _rv_unset ""; _rv_log "no valid bin.cgyro.restart or bin.cgyro.restart.old (tag: $_rv_tv): COLD START, out.cgyro.tag removed$([ -f "$_rv_rs.bad" ] && echo ", blob kept as bin.cgyro.restart.bad")"
    fi
fi
[ "$_rv_apply" = 1 ] || echo "$_rv_t"
