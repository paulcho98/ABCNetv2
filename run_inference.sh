for dataset in "real_benchmark"; do #"real_benchmark"; do
    for model in "stage3_dit4sr_testr_jihye" "stage3_dit4sr_testr_q" "stage2_testr_q"; do
    # for model in "full_dit4sr_baseline_cfg1_vlm" "full_dit4sr_baseline_cfg8_vlm" ; do
        CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
        --eval-only \
        --config-file /home/cvlab08/projects/TAIR_eval/ABCNetv2/configs/BAText/SAM_text_test/ABCNETv2_R_50_TotalText_lexicon.yaml \
        DATASETS.TEST "('${dataset}_${model}',)" \
        OUTPUT_DIR ./outputs/${dataset}_lexicon/${model} \
        TEST.USE_LEXICON True
    done
done