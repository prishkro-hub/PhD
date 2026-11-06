#!/bin/bash
projects=("Loss" "Shift")
stages=("Compute" "Fit")

declare -A folder_map=(
    ["Al_Al2O3"]="6637 6987 7524 8037"
    ["Al_Sapp"]="6427 6789 7308 7835"
    ["Ta_Nb_Sapp"]="6550 6939 7506 8037"
    ["Ta_Sapp"]="6870 7411 8022 8575"
)
for proj in "${projects[@]}"; do
    for mat in "${!folder_map[@]}"; do
        for res in ${folder_map[$mat]}; do
            for stg in "${stages[@]}"; do
                mkdir -p "$proj/Outputs/$mat/${res}_MHz/$stg/Plots"
            done
        done
    done
done
