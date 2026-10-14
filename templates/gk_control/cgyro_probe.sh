_mitim_mt() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null; }
cd @{exec_folder} && for folder in @{folder_list}; do
    info="$folder/out.cgyro.info"; timing="$folder/out.cgyro.timing"; tag="$folder/out.cgyro.tag"
    now=$(date +%s)
    if [ -f "$info" ]; then
        info_mtime=$(_mitim_mt "$info"); wall=$((now - info_mtime))
        if [ -f "$timing" ]; then
            avg=$(awk '/^Run time/{run=1; next} run && NF>=14 && ($NF+0==$NF){s+=$NF; n++} END{if(n>0) printf "%.3f", s/n; else printf "NA"}' "$timing")
            steps=$(awk '/^Run time/{run=1; next} run && NF>=14 && ($NF+0==$NF){n++} END{print n+0}' "$timing")
            timing_mtime=$(_mitim_mt "$timing"); since_update=$((now - timing_mtime))
            state="RUNNING"
        else
            avg="NA"; steps="0"; since_update=$wall; state="INITIALIZED"
        fi
        tag_token="-"
        if [ -f "$tag" ]; then
            tk=$(awk 'NF>0 {print $1; exit}' "$tag")
            [ -n "$tk" ] && tag_token="$tk"
        fi
        exited=$(grep -q "^EXIT" "$info" && echo 1 || echo 0)
        [ -f "$folder/mitim_budget.tag" ] && exited=1
        # what is left of this launch: last TOTAL of out.cgyro.timing [s per a/cs], last simulated time,
        # the time the launch started from (.mitim_t0) and its MAX_TIME (a/cs added on top of t0)
        last=$([ -f "$timing" ] && tail -n1 "$timing" | awk '($NF+0==$NF){print $NF}')
        t_now=$(tail -n1 "$folder/out.cgyro.time" 2>/dev/null | awk '{print $1}')
        t0=$(cat "$folder/.mitim_t0" 2>/dev/null)
        max_time=$(awk -F= '/^[ \t]*MAX_TIME/{gsub(/[ \t]/,"",$2); print $2; exit}' "$folder/input.cgyro" 2>/dev/null)
        echo "$folder|$state|$avg|$steps|$wall|$since_update|$tag_token|$exited|${last:-NA}|${t_now:-NA}|${t0:-NA}|${max_time:-NA}"
    else
        echo "$folder|NOT_STARTED|NA|0|0|0|-|0|NA|NA|NA|NA"
    fi
done
