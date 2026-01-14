#!/bin/bash

images=("grass" "leaf" "pebbles" "sky" "water" "wood")
# images=("sky")

for input in "${images[@]}"; do
    output="${input}"
    python3 main.py --input "data/${input}.jpg" --output "images/${output}.jpg" --epochs 200 --lr 1 --optimizer LBFGS --bf16
done
