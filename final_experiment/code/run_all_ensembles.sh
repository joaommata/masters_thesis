#!/bin/bash
# Run c2_ensemble.py for every (disease, arm, K) that has both ladders finished.
#
#     ./run_all_ensembles.sh                      # the full-table KNN arm, K=1,5,10
#     ARMS="knn:20k diffusion:20k" KS="1 3" ./run_all_ensembles.sh atelectasis
#
# An ARM is "<source>:<subset>", with an empty subset written as "<source>:".
# It has to be spelled out rather than discovered, because the combination that
# is MISSING is exactly the one worth noticing, and a loop over whatever
# directories happen to exist reports nothing when a chain never ran.
cd "$(dirname "$0")" || exit 1

DISEASES=("$@")
if [[ ${#DISEASES[@]} -eq 0 ]]; then
    DISEASES=(effusion atelectasis cardiomegaly consolidation edema)
fi
ARMS="${ARMS:-knn:}"
KS="${KS:-1 5 10}"

for d in "${DISEASES[@]}"; do
    for arm in $ARMS; do
        src="${arm%%:*}"
        sub="${arm##*:}"
        for k in $KS; do
            echo ""
            echo "################  $d  $src${sub:+/$sub}  K=$k  ################"
            python c2_ensemble.py --disease "$d" --cf-source "$src" \
                --cf-count "$k" ${sub:+--subset "$sub"} \
                || echo "  SKIPPED: $d $src${sub:+/$sub} K=$k"
        done
    done
done
