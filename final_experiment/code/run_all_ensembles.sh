#!/bin/bash
# Run c2_ensemble.py for every (disease, K) that has both ladders finished.
cd "$(dirname "$0")" || exit 1

for d in effusion atelectasis cardiomegaly consolidation edema; do
    for k in 1 5 10; do
        echo ""
        echo "################  $d  K=$k  ################"
        python c2_ensemble.py --disease "$d" --cf-source knn --cf-count "$k" \
            || echo "  SKIPPED: $d K=$k"
    done
done
