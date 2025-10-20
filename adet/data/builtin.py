import os

from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .datasets.text import register_text_instances

# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    "totaltext_train": ("totaltext/train_images", "totaltext/train.json"),
    "totaltext_val": ("totaltext/test_images", "totaltext/test.json"),
    "ctw1500_word_train": ("CTW1500/ctwtrain_text_image", "CTW1500/annotations/train_ctw1500_maxlen100_v2.json"),
    "ctw1500_word_test": ("CTW1500/ctwtest_text_image","CTW1500/annotations/test_ctw1500_maxlen100.json"),
    "syntext1_train": ("syntext1/images", "syntext1/annotations/train.json"),
    "syntext2_train": ("syntext2/images", "syntext2/annotations/train.json"),
    "mltbezier_word_train": ("mlt2017/images","mlt2017/annotations/train.json"),
    "rects_train": ("ReCTS/ReCTS_train_images", "ReCTS/annotations/rects_train.json"),
    "rects_val": ("ReCTS/ReCTS_val_images", "ReCTS/annotations/rects_val.json"),
    "rects_test": ("ReCTS/ReCTS_test_images", "ReCTS/annotations/rects_test.json"),
    "art_train": ("ArT/rename_artimg_train", "ArT/annotations/abcnet_art_train.json"), 
    "lsvt_train": ("LSVT/rename_lsvtimg_train", "LSVT/annotations/abcnet_lsvt_train.json"), 
    "chnsyn_train": ("ChnSyn/syn_130k_images", "ChnSyn/annotations/chn_syntext.json"),
    "icdar2013_train": ("icdar2013/train_images", "icdar2013/ic13_train.json"),
    "icdar2015_train": ("icdar2015/train_images", "icdar2015/ic15_train.json"),
    "icdar2015_test": ("icdar2015/test_images", "icdar2015/ic15_test.json"),
    "sam_text_test": ( # This is the name you use in your YAML
        "/media/dataset1/text_restoration/100K/images/test/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_jpg.json" # json_file
    ),
    "sam_text_test_swinir": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/SwinIR/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_real-esrgan": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/Real-ESRGAN/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_jpg.json" # json_file
    ),
    "sam_text_test_resshift": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/ResShift/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_stablesr": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/StableSR/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_diffbirv2": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/DiffBIRv2/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_diffbirv21": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/DiffBIRv21/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_seesr": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/SeeSR/sample00/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_jihye_prompt1_caption": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt1_CAPTIONstyle/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_jihye_prompt1_tag": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt1_TAGstyle", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_jihye_prompt2_caption": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt2_CAPTIONstyle", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_jihye_prompt2_tag": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt2_TAGstyle", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_pho_prompt1_caption": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/PHO_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt1_CAPTIONstyle/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_pho_prompt1_tag": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/PHO_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt1_TAGstyle/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_pho_prompt2_caption": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/PHO_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt2_CAPTIONstyle/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_pho_prompt2_tag": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/PHO_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt2_TAGstyle/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_jihye_prompt0_caption": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_jihye_prompt0_tag": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_TAGstyle", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_pho_prompt0_caption": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/PHO_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_pho_prompt0_tag": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/PHO_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_TAGstyle/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_lq_upscaled": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/generated_data/SAM_text_test_LQ_upscaled", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_jpg.json" # json_file
    ),
    "sam_text_test_jihye_GT_tag": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_TAGstyle", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_pho_GT_tag": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/SAM_text_test/PHO_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_TAGstyle", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    # --- REAL_BENCHMARK VARIANTS ---
    "real_benchmark_hq": ( # Assuming this is the original/HQ for real_benchmark
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/HQ", # Base image_root for real_benchmark HQ
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_jpg.json" # Annotation file for real_benchmark
    ),
    "real_benchmark_lq_upscaled": (
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full_LQ_upscaled", # EXAMPLE path for upscaled LQ
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_jpg.json"
    ),
    "real_benchmark_swinir": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/SwinIR/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json" # Assuming PNG outputs for SwinIR
    ),
    "real_benchmark_real-esrgan": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/Real-ESRGAN/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_jpg.json" # Assuming JPG outputs for Real-ESRGAN
    ),
    "real_benchmark_resshift": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/ResShift/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stablesr": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/StableSR/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_diffbirv2": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/DiffBIRv2/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_diffbirv21": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/DiffBIRv21/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_seesr": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/SeeSR/sample00/", # Assuming similar structure
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_prompt1_caption": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt1_CAPTIONstyle/", # Adjusted base path
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_prompt1_tag": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt1_TAGstyle/", # Adjusted base path
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_prompt2_caption": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt2_CAPTIONstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_prompt2_tag": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt2_TAGstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_prompt0_caption": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_prompt0_tag": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_TAGstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_GT_tag": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_TAGstyle/", # Note: GTrompt vs GTprompt
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_GT_caption": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_CAPTIONstyle/", # Note: GTrompt vs GTprompt
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    # --- REAL_BENCHMARK - LEVEL 1 ---
    "real_benchmark_lv1_lq_upscaled": (
        "/media/dataset1/hyunbin/generated_data/realbenchmark_degradation/lv1_LQ_upscaled",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_jpg.json"
    ),
    "real_benchmark_lv1_swinir": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv1/SwinIR/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv1_real-esrgan": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv1/Real-ESRGAN/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_jpg.json"
    ),
    "real_benchmark_lv1_resshift": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv1/ResShift/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv1_stablesr": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv1/StableSR/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv1_diffbirv2": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv1/DiffBIRv2/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv1_diffbirv21": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv1/DiffBIRv21/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv1_seesr": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv1/SeeSR/sample00/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv1_jihye_prompt0_caption": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv1/val_pho/realbench_lv1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv1_jihye_prompt0_tag": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv1/val_pho/realbench_lv1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_TAGstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv1_jihye_GT_caption": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv1/val_pho/realbench_lv1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_CAPTIONstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv1_jihye_GT_tag": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv1/val_pho/realbench_lv1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_TAGstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),

    # --- REAL_BENCHMARK - LEVEL 2 ---
    "real_benchmark_lv2_lq_upscaled": (
        "/media/dataset1/hyunbin/generated_data/realbenchmark_degradation/lv2_LQ_upscaled",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_jpg.json"
    ),
    "real_benchmark_lv2_swinir": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv2/SwinIR/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv2_real-esrgan": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv2/Real-ESRGAN/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_jpg.json"
    ),
    "real_benchmark_lv2_resshift": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv2/ResShift/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv2_stablesr": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv2/StableSR/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv2_diffbirv2": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv2/DiffBIRv2/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv2_diffbirv21": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv2/DiffBIRv21/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv2_seesr": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv2/SeeSR/sample00/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv2_jihye_prompt0_caption": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv2/val_pho/realbench_lv2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv2_jihye_prompt0_tag": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv2/val_pho/realbench_lv2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_TAGstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv2_jihye_GT_caption": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv2/val_pho/realbench_lv2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_CAPTIONstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv2_jihye_GT_tag": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv2/val_pho/realbench_lv2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_TAGstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),

    # --- REAL_BENCHMARK - LEVEL 3 ---
    "real_benchmark_lv3_lq_upscaled": (
        "/media/dataset1/hyunbin/generated_data/realbenchmark_degradation/lv3_LQ_upscaled",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_jpg.json"
    ),
    "real_benchmark_lv3_swinir": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv3/SwinIR/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv3_real-esrgan": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv3/Real-ESRGAN/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_jpg.json"
    ),
    "real_benchmark_lv3_resshift": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv3/ResShift/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv3_stablesr": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv3/StableSR/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv3_diffbirv2": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv3/DiffBIRv2/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv3_diffbirv21": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv3/DiffBIRv21/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv3_seesr": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv3/SeeSR/sample00/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv3_jihye_prompt0_caption": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv3/val_pho/realbench_lv3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv3_jihye_prompt0_tag": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv3/val_pho/realbench_lv3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_TAGstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv3_jihye_GT_caption": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv3/val_pho/realbench_lv3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_CAPTIONstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_lv3_jihye_GT_tag": (
        "/media/dataset1/hyunbin/benchmarks/realbench_lv3/val_pho/realbench_lv3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_TAGstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    # --- SAMText_test_lv1 VARIANTS ---
    "samtext_lv1_lq_upscaled": (
        "/media/dataset1/hyunbin/generated_data/SAMText_test_degradation/lv1_LQ_upscaled", # Path for upscaled LQ for lv1
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_jpg.json" # Original SAM_text_test annotations
    ),
    "samtext_lv1_swinir": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/SwinIR/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_real-esrgan": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/Real-ESRGAN/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_jpg.json"
    ),
    "samtext_lv1_resshift": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/ResShift/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_stablesr": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/StableSR/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_diffbirv2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/DiffBIRv2/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_diffbirv21": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/DiffBIRv21/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_seesr": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/SeeSR/sample00/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_jihye_prompt0_caption": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v21/samtext_LV1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_jihye_prompt0_tag": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v21/samtext_LV1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_TAGstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_jihye_GT_caption": ( # Assuming GT means prompt0 for this context based on your example
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v21/samtext_LV1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_jihye_GT_tag": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v21/samtext_LV1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_TAGstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),

    # --- SAMText_test_lv2 VARIANTS ---
    "samtext_lv2_lq_upscaled": (
        "/media/dataset1/hyunbin/generated_data/SAMText_test_degradation/lv2_LQ_upscaled",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_jpg.json"
    ),
    "samtext_lv2_swinir": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/SwinIR/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_real-esrgan": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/Real-ESRGAN/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_jpg.json"
    ),
    "samtext_lv2_resshift": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/ResShift/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_stablesr": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/StableSR/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_diffbirv2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/DiffBIRv2/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_diffbirv21": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/DiffBIRv21/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_seesr": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/SeeSR/sample00/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_prompt0_caption": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21/samtext_LV2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_prompt0_tag": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21/samtext_LV2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_TAGstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_GT_caption": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21/samtext_LV2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_GT_tag": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21/samtext_LV2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_TAGstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),

    # --- SAMText_test_lv3 VARIANTS ---
    "samtext_lv3_lq_upscaled": (
        "/media/dataset1/hyunbin/generated_data/SAMText_test_degradation/lv3_LQ_upscaled",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_jpg.json"
    ),
    "samtext_lv3_swinir": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/SwinIR/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_real-esrgan": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/Real-ESRGAN/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_jpg.json"
    ),
    "samtext_lv3_resshift": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/ResShift/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_stablesr": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/StableSR/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_diffbirv2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/DiffBIRv2/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_diffbirv21": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/DiffBIRv21/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_seesr": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/SeeSR/sample00/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_prompt0_caption": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v21/samtext_LV3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_prompt0_tag": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v21/samtext_LV3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_TAGstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_GT_caption": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v21/samtext_LV3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_GT_tag": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v21/samtext_LV3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_TAGstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    
    # DiffBIRv21
    "sam_text_test_diffbirv21_no_prompt": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/DiffBIRv21_no_prompt/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "real_benchmark_diffbirv21_no_prompt": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/DiffBIRv21_no_prompt/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv1_diffbirv21_no_prompt": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/DiffBIRv21_no_prompt/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_diffbirv21_no_prompt": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/DiffBIRv21_no_prompt/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_diffbirv21_no_prompt": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/DiffBIRv21_no_prompt/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    # New models (V2)
    "samtext_lv1_jihye_GT_tag_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v2/samtext_LV1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTrompt_TAGstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_jihye_prompt0_tag_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v2/samtext_LV1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_TAGstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_GT_tag_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v2/samtext_LV2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTrompt_TAGstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_prompt0_tag_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v2/samtext_LV2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_TAGstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_GT_tag_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v2/samtext_LV3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTrompt_TAGstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_prompt0_tag_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v2/samtext_LV3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_TAGstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "sam_text_test_jihye_GT_tag_v2": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/real_real_v2/JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTrompt_TAGstyle/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_jihye_prompt0_tag_v2": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/real_real_v2/JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_TAGstyle/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "real_benchmark_jihye_GT_tag_v2": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v2/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTrompt_TAGstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_prompt0_tag_v2": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v2/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_TAGstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    # V2 CAPTION
    "samtext_lv1_jihye_GT_caption_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v2/samtext_LV1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTrompt_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_jihye_prompt0_caption_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v2/samtext_LV1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_GT_caption_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v2/samtext_LV2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTrompt_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_prompt0_caption_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v2/samtext_LV2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_GT_caption_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v2/samtext_LV3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTrompt_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_prompt0_caption_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v2/samtext_LV3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "sam_text_test_jihye_GT_caption_v2": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/real_real_v2/JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTrompt_CAPTIONstyle/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_jihye_prompt0_caption_v2": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/real_real_v2/JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_CAPTIONstyle/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "real_benchmark_jihye_GT_caption_v2": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v2/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTrompt_CAPTIONstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_prompt0_caption_v2": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v2/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_CAPTIONstyle/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    # SUPIR
    "sam_text_test_supir": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/SUPIR/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "real_benchmark_supir": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/SUPIR/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv1_supir": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/SUPIR/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_supir": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/SUPIR/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_supir": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/SUPIR/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    # FaithDiff
    "sam_text_test_faithdiff": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/FaithDiff/", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "real_benchmark_faithdiff": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/FaithDiff/",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv1_faithdiff": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/FaithDiff/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_faithdiff": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/FaithDiff/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_faithdiff": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/FaithDiff/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    # V2 NULL
    "samtext_lv1_jihye_null_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v2/samtext_LV1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_NULLprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_null_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v2/samtext_LV2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_NULLprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_null_v2": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v2/samtext_LV3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_NULLprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    # "sam_text_test_jihye_null_v2": ( # This is the name you use in your YAML
    #     "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/real_real_v2/JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_CAPTIONstyle/", # image_root
    #     "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    # ),
    # "real_benchmark_jihye_null_v2": (
    #     "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v2/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTrompt_CAPTIONstyle/",
    #     "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    # ),

    # v2 FINAL SAM lvl
    "samtext_lv1_jihye_GT_tag_v2_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v2_final/samtext_LV1_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTrompt_TAGstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_GT_tag_v2_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v2_final/samtext_LV2_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTprompt_TAGstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_GT_tag_v2_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v2_final/samtext_LV3_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTprompt_TAGstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_jihye_prompt0_tag_v2_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v2_final/samtext_LV1_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_TAGstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_prompt0_tag_v2_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v2_final/samtext_LV2_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_TAGstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_prompt0_tag_v2_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v2_final/samtext_LV3_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_TAGstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    # V2 real benchmark
    "real_benchmark_jihye_GT_tag_v2_final": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v2_final/realbench_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTprompt_TAGstyle",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_GT_caption_v2_final": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v2_final/realbench_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_GTprompt_CAPTIONstyle",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_null_v2_final": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v2_final/realbench_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_NULLprompt",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_prompt0_tag_v2_final": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v2_final/realbench_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_TAGstyle",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_prompt0_caption_v2_final": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v2_final/realbench_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV2_unetV2_OCRprompt0_CAPTIONstyle",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    # V2.1 MODELS:
    "samtext_lv1_jihye_prompt0_caption_v21_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v21_final/samtext_LV1_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_jihye_GT_caption_v21_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v21_final/samtext_LV1_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTprompt_CAPTIONstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_prompt0_caption_v21_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21_final/samtext_LV2_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_GT_caption_v21_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21_final/samtext_LV2_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTprompt_CAPTIONstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_prompt0_caption_v21_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v21_final/samtext_LV3_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_GT_caption_v21_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v21_final/samtext_LV3_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTprompt_CAPTIONstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    # V2.1 REAL BENCHMARK + SAMText
    "real_benchmark_jihye_prompt0_caption_v21_final": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v21_final/realbench_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_GT_caption_v21_final": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v21_final/realbench_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_CAPTIONstyle",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "sam_text_test_jihye_prompt0_caption_v21_final": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/real_real_v21_final/samtext_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    "sam_text_test_jihye_GT_caption_v21_final": ( # This is the name you use in your YAML
        "/media/dataset1/hyunbin/benchmarks/SAM_text_test/val_pho/real_real_v21_final/samtext_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_CAPTIONstyle", # image_root
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json" # json_file
    ),
    # Others:
    "samtext_lv2_jihye_prompt0_caption_v21_final_real_ours": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21_final/samtext_LV2_JIHYE_FINAL_STAGE3_swinReal_kernelOurs_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_prompt0_caption_v21_final_code_ours": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21_final/samtext_LV2_JIHYE_FINAL_STAGE3_swinCode_kernelOurs_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_GT_caption_v21_final_stage1": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21_final/samtext_LV2_JIHYE_FINAL_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_GTprompt_CAPTIONstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_GT_tag_v21_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21_final/samtext_LV2_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_NULLprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_prompt0_tag_v21_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21_final/samtext_LV2_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_TAGstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_null_v21_final": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21_final/samtext_LV2_JIHYE_FINAL_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_NULLprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),  
    "real_benchmark_jihye_prompt0_caption": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v21/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_null_v21_stage1": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21/samtext_LV2_JIHYE_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_Nullprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),

    # Ablation
    "samtext_lv2_jihye_null_v21": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21/samtext_LV2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_NULLprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_GT_caption_v21_stage1": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21/samtext_LV2_JIHYE_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_GTprompt_CAPTIONstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_jihye_null_v21_stage1": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v21/samtext_LV1_JIHYE_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_NULLprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json",
    ),
    "samtext_lv1_jihye_GT_caption_v21_stage1": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v21/samtext_LV1_JIHYE_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_GTrompt_CAPTIONstyle/",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json",
    ),
    "samtext_lv1_jihye_null_v21": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v21/samtext_LV1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_NULLprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_null_v21_stage1": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v21/samtext_LV3_JIHYE_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_NULLprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_GT_caption_v21_stage1": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v21/samtext_LV3_JIHYE_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_GTprompt_CAPTIONstyle",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_null_v21": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v21/samtext_LV3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_NULLprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_null_v21": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v21/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_NULLprompt",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_GT_caption_v21": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v21/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_GTprompt_CAPTIONstyle",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_null_v21_stage1": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v21/realbench_JIHYE_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_NULLprompt",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_GT_caption_v21_stage1": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v21/realbench_JIHYE_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_GTprompt_CAPTIONstyle",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_vlm_lq_correction_true": (
        "/data/hyunbin/dataset1/hyunbin/tair_extension/realtext/terediff_stage3/pho_server8_gpu0_RealText_VAL_terediff_stage3_infSampleStep50_qwenVL3B_vlmInputLQ_vlmstep50_vlmCorrectionTrue",
        "/data/hyunbin/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_vlm_clean_correction_false": (
        "/data/hyunbin/dataset1/hyunbin/tair_extension/realtext/terediff_stage3/pho_server8_gpu1_RealText_VAL_terediff_stage3_infSampleStep50_qwenVL3B_vlmInputCLEAN_vlmstep50_vlmCorrectionFalse",
        "/data/hyunbin/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_vlm_lq_correction_false": (
        "/data/hyunbin/dataset1/hyunbin/tair_extension/realtext/terediff_stage3/pho_server8_gpu2_RealText_VAL_terediff_stage3_infSampleStep50_qwenVL3B_vlmInputLQ_vlmstep50_vlmCorrectionFalse",
        "/data/hyunbin/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_vlm_none": (
        "/data/hyunbin/dataset1/hyunbin/tair_extension/realtext/terediff_stage3/pho_server8_gpu3_RealText_VAL_terediff_stage3_infSampleStep50",
        "/data/hyunbin/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_vlm_clean_correction_true": (
        "/data/hyunbin/dataset1/hyunbin/tair_extension/realtext/terediff_stage3/pho_server8_gpu3_RealText_VAL_terediff_stage3_infSampleStep50_qwenVL3B_vlmInputCLEAN_vlmstep50_vlmCorrectionTrue",
        "/data/hyunbin/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_vlm_restore_step2_correction_false": (
        "/data/jinlovespho/iclr2026/tair_extension/realtext/terediff_stage3/pho_server8_gpu0_RealText_VAL_terediff_stage3_infSampleStep50_qwenVL3B_vlmInputRESTORE_vlmstep2_vlmCorrectionFalse",
        "/data/hyunbin/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_vlm_restore_step50_correction_false": (
        "/data/jinlovespho/iclr2026/tair_extension/realtext/terediff_stage3/pho_server8_gpu1_RealText_VAL_terediff_stage3_infSampleStep50_qwenVL3B_vlmInputRESTORE_vlmstep50_vlmCorrectionFalse",
        "/data/hyunbin/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_vlm_restore_step2_correction_true": (
        "/data/jinlovespho/iclr2026/tair_extension/realtext/terediff_stage3/pho_server8_gpu2_RealText_VAL_terediff_stage3_infSampleStep50_qwenVL3B_vlmInputRESTORE_vlmstep2_vlmCorrectionTrue",
        "/data/hyunbin/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_vlm_restore_step50_correction_true": (
        "/data/jinlovespho/iclr2026/tair_extension/realtext/terediff_stage3/pho_server8_gpu3_RealText_VAL_terediff_stage3_infSampleStep50_qwenVL3B_vlmInputRESTORE_vlmstep50_vlmCorrectionTrue",
        "/data/hyunbin/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_no_prompt_training": (
        "/data/hyunbin/nips25_rebuttal/tair_rebuttal/no_prompt_training/terediff_stage3/realtext/pho_rebuttal_server8_gpu1_RealText_VAL_terediff_stage3_infSampleStep50",
        "/data/hyunbin/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_q": (
        "/data/text_restoration/tair_extension/satext_lv3/dit4sr_q",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "real_benchmark_dit4sr_q": (
        "/data/text_restoration/tair_extension/realtext/dit4sr_q",
        "/data/hyunbin/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_q_llava_gt_prompt": (
        "/data/text_restoration/tair_extension/satext_lv3/dit4sr_q_wllava_gtprompt/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_q_llava_null_prompt": (
        "/data/text_restoration/tair_extension/satext_lv3/dit4sr_q_wllava_nullprompt/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_hq": (
        "/data/text_restoration/100K/test",
        "/data/text_restoration/100K/test/dataset_coco_jpg.json"
    ),
    "samtext_lv3_tair": (
        "/data/hyunbin/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v21/samtext_LV3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_OCRprompt0_CAPTIONstyle",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_q_gtprompt_lremb": (
        "/data/text_restoration/tair_extension/satext_lv3/dit4sr_q_gtprompt_lremb/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_q_llavaprompt_lremb": (
        "/data/text_restoration/tair_extension/satext_lv3/dit4sr_q_llavaprompt_lremb/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_q_nullprompt_lremb": (
        "/data/text_restoration/tair_extension/satext_lv3/dit4sr_q_nullprompt_lremb/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_q_qwen3_prompt": (
        "/data/text_restoration/tair_extension/satext_lv3/dit4sr_q_qwen3prompt/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_q_qwen7_prompt": (
        "/data/text_restoration/tair_extension/satext_lv3/dit4sr_q_qwen7prompt/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_q_qwen32_prompt": (
        "/data/text_restoration/tair_extension/satext_lv3/dit4sr_q_qwen32prompt/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_q_qwen72_prompt": (
        "/data/text_restoration/tair_extension/satext_lv3/dit4sr_q_qwen72prompt/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_llava_prompt": (
        "/data/text_restoration/tair_extension/satext_lv3/tair_dit4sr_llavaprompt/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_llava_prompt_ckpt12000": (
        "/data/text_restoration/tair_extension/satext_lv3/tair_dit4sr_llavaprompt_ckpt12000/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_llava_prompt_all_ckpt10000": (
        "/data/text_restoration/tair_extension/satext_lv3/tair_dit4sr_llavaprompt_all_ckpt10000/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_qwen7_prompt_all_ckpt10000": (
        "/data/text_restoration/tair_extension/satext_lv3/dit4sr_q_qwen7prompt_ckpt10000/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr_q_qwen7_prompt_ckpt26000": (
        "/data/text_restoration/tair_extension/satext_lv3/dit4sr_q_qwen7prompt_ckpt26000/sample00",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_qwen7b_prompt": (
        "/data/text_restoration/tair_extension/satext_lv3/tair_qwen7bprompt/pho_server8_gpu0_SATextLv3_VAL_terediff_stage3_infSampleStep50",
        "/data/text_restoration/100K/test/dataset_coco_png.json"
    )

}


metadata_text = {
    "thing_classes": ["text"]
}


def register_all_coco(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


register_all_coco()
