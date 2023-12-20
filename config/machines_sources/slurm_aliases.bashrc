# Created with the use of GPT-4

# --------------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------------

alias sl='python3 $MITIM_PATH/mitim_opt/opt_tools/exe/slurm.py $1 --partition $MITIM_PARTITION $2'

# --------------------------------------------------------------------------------
# Information
# --------------------------------------------------------------------------------

alias sa='salloc --partition $MITIM_PARTITION --nodes=1 --exclusive --time=8:00:00'

export SACCT_FORMAT="Partition%22,User,JobID%17,JobName%25,NodeList%15,Elapsed,State%10,CPUTime%15,AllocTRES%40"

export SINFO_FORMAT="%21P %5c %10m %10a %10l %6D %10t %30g"
alias si='sinfo -p $MITIM_PARTITIONS_ALL | awk -v partitions="$MITIM_PARTITIONS_ALL" "BEGIN {OFS=\"\t\"; split(\"31:32:34:33:35:36\", colors, \":\"); split(partitions, parts, \",\"); for (i in parts) colorMap[parts[i]] = colors[++idx]; } NR==1 {print; next} {printf \"\033[%sm%s\033[0m\n\", colorMap[\$1], \$0}"'
alias si_idle='(si | head -n 1; si | grep -w "idle")'

alias myqos="sacctmgr show qos format=Name%30,Priority,MaxWall,MaxJobsPU"

export SQUEUE_FORMAT="%.21P %.7u %.8i %.5D %.4C %.10T %.8M %.24j %.10l %R"

# all my runs
alias sq='squeue -u $USER | awk -v user=$USER \
"BEGIN {
    split(\"31:32:34:35:36:33\", colors, \":\");
}
NR==1 {
    printf \"\033[33m%s\033[0m\n\", \$0;  # Set header to yellow
    next;
}
{
    partition=\$1;
    if (!(partition in colorMap)) {
        colorMap[partition] = colors[++idx % length(colors)];
    }
    gsub(partition, \"\033[\" colorMap[partition] \"m\" partition \"\033[0m\");
    if (index(\$0, user) > 0) {
        gsub(user, \"\033[33m\" user \"\033[0m\");  # Set user highlight to yellow
    }
    print;
}"'
# users using my partition
alias squ="f_jobs_per_user() { \
    echo -e '\033[1;33mThe following users are using the partition\033[0;32m' \$MITIM_PARTITION '\033[0m'; \
    echo -e '\033[0;31mJobs\tUser\033[0m'; \
    squeue -p \$MITIM_PARTITION | awk 'NR>1 {print \$2}' | sort | uniq -c | awk '{printf \"%-6s\\t%s\\n\", \$1, \$2}'; \
    IDLE_NODES_INFO=\$(sinfo -p \$MITIM_PARTITION --noheader -h | awk '\$7 == \"idle\" {print \$6 \"x\" \$2}'); \
    echo -e '\033[1;33m-------------------------------------\033[0m'; \
    echo -e '\033[1;33mIdle nodes info (nodes x cpus):\033[0;32m' \$IDLE_NODES_INFO '\033[0m'; \
    unset -f f_jobs_per_user; \
}; f_jobs_per_user"
# all jobs in my partition
alias sqe='squeue -p $MITIM_PARTITION | awk -v partition=$MITIM_PARTITION \
"BEGIN {
    split(\"31:32:34:35:36:33\", colors, \":\");
}
NR==1 {
    printf \"\033[31m%s\033[0m\n\", \$0;
    next;
}
{
    user=\$2;  # Assuming that the user is the second field in your squeue format
    if (!(user in colorMap)) {
        colorMap[user] = colors[++idx % length(colors)];
    }
    gsub(user, \"\033[\" colorMap[user] \"m\" user \"\033[0m\");
    print;
}"'

# --------------------------------------------------------------------------------
# Cancellations
# --------------------------------------------------------------------------------

# Removes all jobs from user
alias sc_all='scancel -u $USER'

# Removes by job name, e.g. "sc mitim_opt_jet1_s5"
sc() {
    if [ -z "$1" ]; then
        echo "Please provide a job name."
        return 1
    fi
    scancel $(squeue -u $USER | grep $1 | awk '{print $3}')
    echo "Cancelled jobs with name: $1"
}

# Removes all jobs by base name, e.g. "sc_seed mitim_opt_jet1" will remove everything that starts with mitim_opt_jet1
sc_seed() {
    if [ -z "$1" ]; then
        echo "Please provide a base job name."
        return 1
    fi

    scancel $(squeue -u $USER | awk -v base="$1" 'index($8, base) {print $3}')
    echo "Cancelled all jobs with base name: $1"
}

