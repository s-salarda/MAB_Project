#!/bin/bash

###############################################
# USER SETTINGS
###############################################

PAIRFILE="distance_pairs.txt"

PRMTOP_WT="1.m.adp.pi_wt_model1_strip.prmtop"
PRMTOP_MUT="1.m.adp.pi_d239n_model1_strip.prmtop"

TRAJ_WT=(
    "1.m.adp.pi_wt_1_2000ns_1000ps.nc"
    "1.m.adp.pi_wt_2_2000ns_1000ps.nc"
    "1.m.adp.pi_wt_3_2000ns_1000ps.nc"
)

TRAJ_MUT=(
    "1.m.adp.pi_d239n_1_500ns_1000ps.nc"
    "1.m.adp.pi_d239n_2_500ns_1000ps.nc"
    "1.m.adp.pi_d239n_3_500ns_1000ps.nc"
)

OUT_WT="results_wt"
OUT_MUT="results_d239n"

mkdir -p $OUT_WT
mkdir -p $OUT_MUT


###############################################
# FUNCTION: Build cpptraj distance commands
###############################################
build_distance_commands () {
    while IFS= read -r line; do
        [[ "$line" =~ ^# || -z "$line" ]] && continue

        LABEL=$(echo $line | awk '{print $1}')
        A1=$(echo $line | awk '{print $2}')
        A2=$(echo $line | awk '{print $3}')

        echo "distance ${LABEL}_${TAG} $A1 $A2 out ${LABEL}_${TAG}.dat"
    done < "$PAIRFILE"
}


###############################################
# RUN WT TRAJECTORIES
###############################################
echo "Running WT trajectories..."

i=1
for TRAJ in "${TRAJ_WT[@]}"; do
    TAG="wt${i}"
    i=$((i+1))

    DIST_CMDS=$(build_distance_commands)
    CPPIN="$OUT_WT/${TAG}_dist.in"

    cat > "$CPPIN" << EOF
parm $PRMTOP_WT
trajin $TRAJ 1 500

$DIST_CMDS

run
EOF

    cpptraj -i "$CPPIN"
done


###############################################
# RUN D239N TRAJECTORIES
###############################################
echo "Running D239N trajectories..."

i=1
for TRAJ in "${TRAJ_MUT[@]}"; do
    TAG="mut${i}"
    i=$((i+1))

    DIST_CMDS=$(build_distance_commands)
    CPPIN="$OUT_MUT/${TAG}_dist.in"

    cat > "$CPPIN" << EOF
parm $PRMTOP_MUT
trajin $TRAJ 1 500

$DIST_CMDS

run
EOF

    cpptraj -i "$CPPIN"
done

echo "All distance calculations complete."
