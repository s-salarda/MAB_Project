#!/bin/bash

RESFILE="residues.txt"

# WT files
WT_PRMTOP="1.m.adp.pi_wt_model1_strip.prmtop"
WT_TRAJS=(
    "1.m.adp.pi_wt_1_2000ns_1000ps.nc"
    "1.m.adp.pi_wt_2_2000ns_1000ps.nc"
    "1.m.adp.pi_wt_3_2000ns_1000ps.nc"
)

# MUT files
MUT_PRMTOP="1.m.adp.pi_d239n_model1_strip.prmtop"
MUT_TRAJS=(
    "1.m.adp.pi_d239n_1_500ns_1000ps.nc"
    "1.m.adp.pi_d239n_2_500ns_1000ps.nc"
    "1.m.adp.pi_d239n_3_500ns_1000ps.nc"
)

# -----------------------------
# Parse residues.txt
# -----------------------------
parse_residues() {
    local file="$1"
    local current_section=""
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue

        if [[ "$line" =~ ^# ]]; then
            current_section="${line#\# }"
            continue
        fi

        if [[ "$line" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start="${BASH_REMATCH[1]}"
            end="${BASH_REMATCH[2]}"
            for ((r=start; r<=end; r++)); do
                echo "$current_section $r"
            done
            continue
        fi

        if [[ "$line" =~ ^[0-9]+$ ]]; then
            echo "$current_section $line"
            continue
        fi
    done < "$file"
}

# -----------------------------
# Run cpptraj for one simulation
# -----------------------------
run_sim() {
    local label="$1"
    local prmtop="$2"
    local traj="$3"
    local simname="$4"

    mkdir -p "$label/$simname"
    LOGFILE="$label/$simname/commands_used.txt"
    echo "Commands used for $simname" > "$LOGFILE"
    echo "PRMTOP: $prmtop" >> "$LOGFILE"
    echo "TRAJ:   $traj" >> "$LOGFILE"
    echo "" >> "$LOGFILE"

    parse_residues "$RESFILE" | while read SECTION RES; do
        CPP="$label/$simname/${SECTION}_${RES}.in"
        OUT="$label/$simname/${SECTION}_${RES}.dat"

        cat > "$CPP" <<EOF
parm $prmtop
trajin $traj 1 500
energy out $OUT mask :$RES
run
EOF

        echo "cpptraj -i ${SECTION}_${RES}.in" >> "$LOGFILE"
        cpptraj -i "$CPP"
    done
}

# -----------------------------
# Run WT sims
# -----------------------------
i=1
for T in "${WT_TRAJS[@]}"; do
    run_sim "WT" "$WT_PRMTOP" "$T" "WT${i}"
    ((i++))
done

# -----------------------------
# Run MUT sims
# -----------------------------
i=1
for T in "${MUT_TRAJS[@]}"; do
    run_sim "MUT" "$MUT_PRMTOP" "$T" "MUT${i}"
    ((i++))
done
