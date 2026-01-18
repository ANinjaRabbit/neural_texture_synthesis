#!/bin/bash

OUTDIR="images/occur_penalty_test"
mkdir -p "$OUTDIR"

# test_img=("leaf")
# test_img=("wood")
# test_img=("sky" "water" "pebbles" "grass")
# test_img=("rust-small")
test_img=("rust2-small")

for img in "${test_img[@]}"; do
    for penalty in 0.0 0.01 0.05; do
        python main-gc.py \
            --input "data/$img.jpg"\
            --output "$OUTDIR/${img}_${penalty}.png" \
            --coef_occur "${penalty}" \
            --output_size 512 512
    done
done
