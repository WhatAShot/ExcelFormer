#!/bin/bash -i

# Correspond to 'Fully Tuned' step in the paper, we further get 'Fully Tuned' ExcelFormer based on 'Mix Tuned'
# Add provided datasets or private datasets (refer to our provided datasets for data preparation)
DATASETS=('analcatdata_supreme' 'isolet' 'cpu_act' 'visualizing_soil' 'yprop_4_1' 'gesture' 'churn' 'sulfur' 'bank-marketing' 'Brazilian_houses')
# Default normalization method is 'quantile' in this paper, choose other methods in ['standard', 'quantile'] upon your datasets
# We follow FT-Transformer(Borisov et al., 2021) for data preprocessing [https://github.com/yandex-research/tabular-dl-revisiting-models/blob/main/lib/data.py]

for i in "${!DATASETS[@]}"; do
    python tune_fully_after_mix_tune.py \
        --dataset ${DATASETS[$i]}
done
