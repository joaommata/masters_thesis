#!/bin/bash
# Run the ConfidNet baseline (c2_confidnet.py) for every disease, then the Aim 2 comparison
# (c2_confidnet_compare.py) on the saved bootstrap draws.
#
#     ./run_all_confidnet.sh                      # all five diseases, sequentially
#     ./run_all_confidnet.sh effusion edema       # a subset (the comparison is then skipped)
#     EXTRA="--overwrite" ./run_all_confidnet.sh
#
# Sequential and CPU-bound (~1-2 h per disease on 8 cores). On the cluster use
# ../jobs/submit_confidnet.sh instead, which runs the diseases in parallel.
cd "$(dirname "$0")" || exit 1

ALL=(effusion atelectasis cardiomegaly consolidation edema)
DISEASES=("$@")
if [[ ${#DISEASES[@]} -eq 0 ]]; then
    DISEASES=("${ALL[@]}")
fi

failed=0
for d in "${DISEASES[@]}"; do
    echo ""
    echo "################  ConfidNet  $d  ################"
    python -u c2_confidnet.py --disease "$d" --cf-source knn --cf-strategy correct_cf \
        --cf-count 10 ${EXTRA:-} || { echo "  FAILED: $d"; failed=1; }
done

# The comparison replays draws shared by all five diseases, so it needs every one of them.
if [[ $failed -eq 0 && ${#DISEASES[@]} -eq ${#ALL[@]} ]]; then
    echo ""
    echo "################  Aim 2 comparison  ################"
    python -u c2_confidnet_compare.py --overwrite
else
    echo ""
    echo "comparison skipped: needs all five diseases to have finished"
fi
