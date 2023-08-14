#!/bin/bash -i

SEEDS=(42 20 31 53 64)
# Add provided datasets or private datasets (refer to our provided datasets for data preparation)
DATASETS=('analcatdata_supreme' 'isolet' 'cpu_act' 'visualizing_soil' 'yprop_4_1' 'gesture' 'churn' 'sulfur' 'bank-marketing' 'Brazilian_houses')
# Default normalization method is 'quantile' in this paper, choose other methods in ['standard', 'quantile'] upon your datasets
# We follow FT-Transformer(Borisov et al., 2021) for data preprocessing [https://github.com/yandex-research/tabular-dl-revisiting-models/blob/main/lib/data.py]

for SEED in ${SEEDS[@]}; do
    for i in "${!DATASETS[@]}"; do
        python run_default_config_excel.py \
            --seed $SEED \
            --dataset ${DATASETS[$i]} \
            --catenc \
            --mixup \
            --mix_type "feat_mix"
        
        python run_default_config_excel.py \
            --seed $SEED \
            --dataset ${DATASETS[$i]} \
            --catenc \
            --mixup \
            --mix_type "hidden_mix"
    done
done