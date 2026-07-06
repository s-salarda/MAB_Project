#!/bin/bash

###############################################
# INPUT FILES
###############################################

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

OUT_WT="angles_wt"
OUT_MUT="angles_mut"

mkdir -p "$OUT_WT" "$OUT_MUT"

###############################################
# FUNCTION TO WRITE A CPPTRAJ INPUT FILE
###############################################
write_cpptraj_file () {
    local PRMTOP=$1
    local TRAJ=$2
    local TAG=$3
    local OUTDIR=$4
    local CPPIN="$OUTDIR/${TAG}_angles.in"

cat > "$CPPIN" << EOF
parm $PRMTOP
trajin $TRAJ 1 500 1

# Align all Helices
rmsd :417-447,478-503@CA 

# Relay helix vector: C-term -> N-term
vector v_relay minimage :417-429@CA,C,N :429-447@CA,C,N out Relay_vec_${TAG}.dat magnitude

# O helix vector: N-term -> C-term
vector v_ohelix  minimage :479-480@CA,C,N :480-503@CA,C,N out OHelix_vec_${TAG}.dat magnitude


# Angles between helices
vectormath vec1 v_relay vec2 v_ohelix dotangle out Relay_OHelix_${TAG}.dat


go
EOF

    echo "$CPPIN"
}

###############################################
# RUN WT
###############################################
echo "Running WT helix angle analysis..."

i=1
for TRAJ in "${TRAJ_WT[@]}"; do
    TAG="wt${i}"
    CPPIN=$(write_cpptraj_file "$PRMTOP_WT" "$TRAJ" "$TAG" "$OUT_WT")
    cpptraj -i "$CPPIN"
    i=$((i+1))
done

###############################################
# RUN MUT
###############################################
echo "Running MUT helix angle analysis..."

i=1
for TRAJ in "${TRAJ_MUT[@]}"; do
    TAG="mut${i}"
    CPPIN=$(write_cpptraj_file "$PRMTOP_MUT" "$TRAJ" "$TAG" "$OUT_MUT")
    cpptraj -i "$CPPIN"
    i=$((i+1))
done

echo "Helix angle analysis complete."
