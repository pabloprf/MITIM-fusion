# A continuation in place (out.cgyro.tag present: MITIM's interrupted-run rescue or a SLURM requeue) resumes
# at output i_current, line 1 of out.cgyro.tag, written right after that output's records. CGYRO rewinds its
# ASCII files to i_current outputs, but the per-output binaries have been found holding the records written
# after it (509 records against 501 rows of out.cgyro.time), which the reader cannot reshape. Cut each one to
# i_current records before CGYRO starts, as its own rewind would. The record count of the binaries is the
# row count of out.cgyro.time when it divides every one of them, else the count bin.cgyro.ky_flux holds for
# the out.cgyro.grids sizes nearest that row count; when neither divides every binary, nothing is cut.
# Warm starts from another evaluation stage no out.cgyro.tag and are left alone. Files never grow.
_tr_dir="@{run_dir}"
if [ -f "$_tr_dir/out.cgyro.tag" ]; then
    _tr_k=$(sed -n 1p "$_tr_dir/out.cgyro.tag" 2>/dev/null | awk '{print int($1)}')
    _tr_nt=$(awk 'NF' "$_tr_dir/out.cgyro.time" 2>/dev/null | wc -l | awk '{print $1+0}')
    _tr_files=""
    for _tr_f in ky_flux ky_cflux lky_flux_n lky_flux_e lky_flux_v kxky_n kxky_e kxky_v kxky_phi kxky_apar kxky_bpar freq hb phib aparb bparb; do
        [ -s "$_tr_dir/bin.cgyro.$_tr_f" ] && _tr_files="$_tr_files $_tr_dir/bin.cgyro.$_tr_f"
    done
    _tr_size() { wc -c < "$1" | awk '{print $1+0}'; }
    _tr_divides() { local f; for f in $_tr_files; do (( $(_tr_size "$f") % $1 == 0 )) || return 1; done; }
    _tr_n=0
    if [ -n "$_tr_files" ] && (( ${_tr_k:-0} > 0 && _tr_nt > 0 )); then
        if _tr_divides "$_tr_nt"; then
            _tr_n=$_tr_nt
        elif [ -s "$_tr_dir/bin.cgyro.ky_flux" ] && [ -f "$_tr_dir/out.cgyro.grids" ]; then
            # one ky_flux record: n_n x n_species x n_field values x 3 or 4 moments x 4 or 8 bytes
            _tr_base=$(awk '{for (i=1; i<=NF; i++) {n++; if (n<=3) p = (n==1 ? $i : p*$i)}} END {print int(p)}' "$_tr_dir/out.cgyro.grids")
            _tr_s=$(_tr_size "$_tr_dir/bin.cgyro.ky_flux"); _tr_best=-1
            for _tr_r in $(( _tr_base*12 )) $(( _tr_base*16 )) $(( _tr_base*24 )) $(( _tr_base*32 )); do
                (( _tr_r > 0 && _tr_s % _tr_r == 0 )) || continue
                _tr_c=$(( _tr_s / _tr_r )); _tr_d=$(( _tr_c > _tr_nt ? _tr_c - _tr_nt : _tr_nt - _tr_c ))
                if (( _tr_best < 0 || _tr_d < _tr_best )); then _tr_best=$_tr_d; _tr_n=$_tr_c; fi
            done
            (( _tr_n > 0 )) && ! _tr_divides "$_tr_n" && _tr_n=0
        fi
        if (( _tr_n > _tr_k )); then
            for _tr_f in $_tr_files; do
                _tr_keep=$(( $(_tr_size "$_tr_f") / _tr_n * _tr_k ))
                truncate -s "$_tr_keep" "$_tr_f" 2>/dev/null || dd if=/dev/null of="$_tr_f" bs=1 seek="$_tr_keep" 2>/dev/null
            done
            echo "MITIM: continuing in place from output $_tr_k: binaries cut from $_tr_n to $_tr_k records" >&2
        elif (( _tr_n == 0 )); then
            echo "MITIM: continuing in place from output $_tr_k: record count of the binaries unknown, none cut" >&2
        fi
    fi
fi
