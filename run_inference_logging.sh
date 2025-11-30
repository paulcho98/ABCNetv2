# for dataset in "samtext_lv3" ; do #"real_benchmark"; do
#     # for model in "diffbirv2" "faithdiff" "lq_upscaled" "terediff_pred_tsmprompt" "terediff_gtprompt" "terediff_nullprompt" "full_dit4sr_baseline_cfg1_null" "full_dit4sr_baseline_cfg1_vlm" "full_dit4sr_baseline_cfg1_gt" "full_dit4sr_baseline_cfg8_null" "full_dit4sr_baseline_cfg8_vlm" "full_dit4sr_baseline_cfg8_gt"; do
#     # for model in "diffbirv2" "faithdiff" "lq_upscaled" "terediff_pred_tsmprompt" "terediff_gtprompt" "terediff_nullprompt" "full_dit4sr_baseline_cfg1_null" "full_dit4sr_baseline_cfg1_vlm" "full_dit4sr_baseline_cfg1_gt" "full_dit4sr_baseline_cfg8_null" "full_dit4sr_baseline_cfg8_vlm" "full_dit4sr_baseline_cfg8_gt"; do
#     # for model in "stage1_jihye_stage2_dit4sr_40k_gt_prompt" "stage1_jihye_stage2_dit4sr_40k_null_prompt" "stage1_jihye_stage2_q_dit4sr_40k_tsm_prompt" "stage1_jihye_stage2_dit4sr_40k_tsm_prompt"; do
#     for model in "stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10__cfg1" "stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10-20__cfg1" "stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20__cfg1" "stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-25-30-35__cfg1" "stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-30__cfg1" "stage2__FINAL-vlm-ques1-qwenvl-7b-correct-30__cfg1"; do
#         EVAL_SAVE_DETAILS=1 EVAL_DETAILS_DIR=details/${dataset}/${model} CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
#         --eval-only \
#         --config-file /home/cvlab08/projects/TAIR_eval/ABCNetv2/configs/BAText/SAM_text_test/ABCNETv2_R_50_TotalText_lexicon.yaml \
#         DATASETS.TEST "('${dataset}_${model}',)" \
#         OUTPUT_DIR ./outputs/${dataset}_lexicon/${model} \
#         TEST.USE_LEXICON True
#     done
# done

for dataset in "samtext_lv1" "samtext_lv2" "samtext_lv3"; do #"real_benchmark"; do
    # for model in "diffbirv2" "faithdiff" "lq_upscaled" "terediff_pred_tsmprompt" "terediff_gtprompt" "terediff_nullprompt" "full_dit4sr_baseline_cfg1_null" "full_dit4sr_baseline_cfg1_vlm" "full_dit4sr_baseline_cfg1_gt" "full_dit4sr_baseline_cfg8_null" "full_dit4sr_baseline_cfg8_vlm" "full_dit4sr_baseline_cfg8_gt"; do
    # for model in "diffbirv2" "faithdiff" "lq_upscaled" "terediff_pred_tsmprompt" "terediff_gtprompt" "terediff_nullprompt" "full_dit4sr_baseline_cfg1_null" "full_dit4sr_baseline_cfg1_vlm" "full_dit4sr_baseline_cfg1_gt" "full_dit4sr_baseline_cfg8_null" "full_dit4sr_baseline_cfg8_vlm" "full_dit4sr_baseline_cfg8_gt"; do
    # for model in "stage1_jihye_stage2_dit4sr_40k_gt_prompt" "stage1_jihye_stage2_dit4sr_40k_null_prompt" "stage1_jihye_stage2_q_dit4sr_40k_tsm_prompt" "stage1_jihye_stage2_dit4sr_40k_tsm_prompt"; do
    for model in "dreamclear"; do
        EVAL_SAVE_DETAILS=1 EVAL_DETAILS_DIR=details/${dataset}/${model} CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
        --eval-only \
        --config-file /home/cvlab08/projects/TAIR_eval/ABCNetv2/configs/BAText/SAM_text_test/ABCNETv2_R_50_TotalText_lexicon.yaml \
        DATASETS.TEST "('${dataset}_${model}',)" \
        OUTPUT_DIR ./outputs/${dataset}_lexicon/${model} \
        TEST.USE_LEXICON True
    done
done

# for dataset in "real_benchmark"; do #"real_benchmark"; do
#     # for model in "diffbirv2" "faithdiff" "lq_upscaled" "terediff_pred_tsmprompt" "terediff_gtprompt" "terediff_nullprompt" "full_dit4sr_baseline_cfg1_null" "full_dit4sr_baseline_cfg1_vlm" "full_dit4sr_baseline_cfg1_gt" "full_dit4sr_baseline_cfg8_null" "full_dit4sr_baseline_cfg8_vlm" "full_dit4sr_baseline_cfg8_gt"; do
#     # for model in "diffbirv2" "faithdiff" "lq_upscaled" "terediff_pred_tsmprompt" "terediff_gtprompt" "terediff_nullprompt" "full_dit4sr_baseline_cfg1_null" "full_dit4sr_baseline_cfg1_vlm" "full_dit4sr_baseline_cfg1_gt" "full_dit4sr_baseline_cfg8_null" "full_dit4sr_baseline_cfg8_vlm" "full_dit4sr_baseline_cfg8_gt"; do
#     # for model in "stage1_jihye_stage2_dit4sr_40k_gt_prompt" "stage1_jihye_stage2_dit4sr_40k_null_prompt" "stage1_jihye_stage2_q_dit4sr_40k_tsm_prompt" "stage1_jihye_stage2_dit4sr_40k_tsm_prompt"; do
#     for model in "dreamclear"; do
#         EVAL_SAVE_DETAILS=1 EVAL_DETAILS_DIR=details/${dataset}/${model} CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
#         --eval-only \
#         --config-file /home/cvlab08/projects/TAIR_eval/ABCNetv2/configs/BAText/SAM_text_test/ABCNETv2_R_50_TotalText_lexicon.yaml \
#         DATASETS.TEST "('${dataset}_${model}',)" \
#         OUTPUT_DIR ./outputs/${dataset}_lexicon/${model} \
#         TEST.USE_LEXICON True
#     done
# done