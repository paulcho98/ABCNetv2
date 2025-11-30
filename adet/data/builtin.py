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
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/HQ",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_jpg.json" # Annotation file for real_benchmark
    ),
    "real_benchmark_lq_upscaled": (
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full_LQ_upscaled", # EXAMPLE path for upscaled LQ
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_jpg.json"
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
        "/home/cvlab08/projects/data2/text_restoration/benchmarks/real_benchmark/DiffBIRv2/",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
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
        "vscode-remote://ssh-remote%2Bcvlab12/media/dataset1/hyunbin/generated_data/realbenchmark_degradation/lv1_LQ_upscaled",
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
        "vscode-remote://ssh-remote%2Bcvlab12/media/dataset1/hyunbin/generated_data/realbenchmark_degradation/lv2_LQ_upscaled",
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
        "vscode-remote://ssh-remote%2Bcvlab12/media/dataset1/hyunbin/generated_data/realbenchmark_degradation/lv3_LQ_upscaled",
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
        "/home/cvlab08/projects/data2/text_restoration/SAMText_test_degradation/lv1_LQ_upscaled", # Path for upscaled LQ for lv1
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_jpg.json" # Original SAM_text_test annotations
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
        "/home/cvlab08/projects/data2/text_restoration/benchmarks/SAMText_test_lv1/DiffBIRv2/",
        '/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json'
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
        "/home/cvlab08/projects/data2/text_restoration/SAMText_test_degradation/lv2_LQ_upscaled",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_jpg.json"
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
        "/home/cvlab08/projects/data2/text_restoration/benchmarks/SAMText_test_lv2/DiffBIRv2/",
        '/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json'
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
        "/home/cvlab08/projects/data2/text_restoration/SAMText_test_degradation/lv3_LQ_upscaled",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_jpg.json"
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
        "/home/cvlab08/projects/data2/text_restoration/benchmarks/SAMText_test_lv3/DiffBIRv2/",
        '/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json'
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
        "/home/cvlab08/projects/data2/text_restoration/benchmarks/real_benchmark/FaithDiff",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv1_faithdiff": (
        "/home/cvlab08/projects/data2/text_restoration/benchmarks/SAMText_test_lv1/FaithDiff/",
        '/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json'
    ),
    "samtext_lv2_faithdiff": (
        "/home/cvlab08/projects/data2/text_restoration/benchmarks/SAMText_test_lv2/FaithDiff/",
        '/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json'
    ),
    "samtext_lv3_faithdiff": (
        "/home/cvlab08/projects/data2/text_restoration/benchmarks/SAMText_test_lv3/FaithDiff/",
        '/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json'
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
    # LLAVA
    "real_benchmark_jihye_llava_v21_stage1": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v21/realbench_JIHYE_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_LLAVAprompt",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_jihye_llava_v21": (
        "/media/dataset1/hyunbin/benchmarks/real_benchmark/val_pho/real_real_v21/realbench_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_LLAVAprompt",
        "/media/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv1_jihye_llava_v21": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v21/samtext_LV1_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_LLAVAprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_llava_v21": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21/samtext_LV2_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_LLAVAprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_llava_v21": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v21/samtext_LV3_JIHYE_STAGE3_swinReal_kernelReal_ctrlV21_unetV21_LLAVAprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv1_jihye_llava_v21_stage1": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv1/val_pho/real_real_v21/samtext_LV1_JIHYE_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_LLAVAprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv2_jihye_llava_v21_stage1": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv2/val_pho/real_real_v21/samtext_LV2_JIHYE_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_LLAVAprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
    ),
    "samtext_lv3_jihye_llava_v21_stage1": (
        "/media/dataset1/hyunbin/benchmarks/SAMText_test_lv3/val_pho/real_real_v21/samtext_LV3_JIHYE_STAGE1_swinReal_kernelReal_ctrlV21_unetV21_LLAVAprompt",
        "/media/dataset1/text_restoration/100K/images/test/dataset_coco_png.json"
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
    ),"real_benchmark_vlm_restore_step2_correction_true": (
        "/data/jinlovespho/iclr2026/tair_extension/realtext/terediff_stage3/pho_server8_gpu2_RealText_VAL_terediff_stage3_infSampleStep50_qwenVL3B_vlmInputRESTORE_vlmstep2_vlmCorrectionTrue",
        "/data/hyunbin/dataset1/hyunbin/generated_data/real_benchmark_full/dataset_coco_png.json"
    ),"real_benchmark_vlm_restore_step50_correction_true": (
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
        "/home/cvlab08/projects/data2/text_restoration/100K/test",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_jpg.json"
    ),
    "samtext_lv2_hq": (
        "/home/cvlab08/projects/data2/text_restoration/100K/test",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_jpg.json"
    ),
    "samtext_lv1_hq": (
        "/home/cvlab08/projects/data2/text_restoration/100K/test",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_jpg.json"
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
    ),
    "real_benchmark_terediff_pred_tsmprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_terediff/realtext/terediff_pred_tsmprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_terediff_gtprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_terediff/realtext/terediff_gtprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_terediff_nullprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_terediff/realtext/terediff_nullprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_dit4sr_baseline_nullprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr/realtext/dit4sr_baseline_lrstartpoint_nullprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_dit4sr_baseline_vlmprompt_llava7b_ques3": (
        '/home/cvlab08/projects/data/tair_result/result_val_dit4sr/realtext/dit4sr_baseline_lrstartpoint_pred_vlmprompt_llava_7b_ques3_/final_restored_img',
        '/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json'
    ),
    "real_benchmark_dit4sr_baseline_gtprompt": (
        '/home/cvlab08/projects/data/tair_result/result_val_dit4sr/realtext/dit4sr_baseline_lrstartpoint_gtprompt/final_restored_img',
        '/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json'
    ),
    "real_benchmark_stage3_dit4sr_testr_5e6_5e5_nullprompt": (
        '/home/cvlab08/projects/data/tair_result/result_val_dit4sr/realtext/fp16_stage3_dit4sr-testr_5e-06-5e-05_lrbranch-attns_ocrloss0.01_descriptive_DiTfeat24_lrstartpoint_nullprompt_/final_restored_img',
        '/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json'
    ),
    "real_benchmark_stage3_dit4sr_testr_5e6_5e5_tsmprompt": (
        '/home/cvlab08/projects/data/tair_result/result_val_dit4sr/realtext/fp16_stage3_dit4sr-testr_5e-06-5e-05_lrbranch-attns_ocrloss0.01_descriptive_DiTfeat24_lrstartpoint_pred_tsmprompt_/final_restored_img',
        '/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json'
    ),
    "real_benchmark_stage3_dit4sr_testr_5e6_5e5_gtprompt": (
        '/home/cvlab08/projects/data/tair_result/result_val_dit4sr/realtext/fp16_stage3_dit4sr-testr_5e-06-5e-05_lrbranch-attns_ocrloss0.01_descriptive_DiTfeat24_lrstartpoint_gtprompt_/final_restored_img',
        '/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json'
    ),
    # SaText lv2 
    "samtext_lv2_dit4sr_baseline_nullprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr/satext_lv2/dit4sr_baseline_lrstartpoint_nullprompt_/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_dit4sr_baseline_vlmprompt_llava7b_ques3": (
        '/home/cvlab08/projects/data/tair_result/result_val_dit4sr/satext_lv2/dit4sr_baseline_lrstartpoint_pred_vlmprompt_llava_7b_ques3_/final_restored_img',
        '/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json'
    ),
    "samtext_lv2_dit4sr_baseline_gtprompt": (
        '/home/cvlab08/projects/data/tair_result/result_val_dit4sr/satext_lv2/dit4sr_baseline_lrstartpoint_gtprompt_/final_restored_img',
        '/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json'
    ),
    "samtext_lv2_stage3_dit4sr_testr_5e6_5e5_nullprompt": (
        '/home/cvlab08/projects/data/tair_result/result_val_dit4sr/satext_lv2/fp16_stage3_dit4sr-testr_5e-06-5e-05_lrbranch-attns_ocrloss0.01_descriptive_DiTfeat24_lrstartpoint_nullprompt_/final_restored_img',
        '/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json'
    ),
    "samtext_lv2_stage3_dit4sr_testr_5e6_5e5_tsmprompt": (
        '/home/cvlab08/projects/data/tair_result/result_val_dit4sr/satext_lv2/fp16_stage3_dit4sr-testr_5e-06-5e-05_lrbranch-attns_ocrloss0.01_descriptive_DiTfeat24_lrstartpoint_pred_tsmprompt_/final_restored_img',
        '/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json'
    ),
    "samtext_lv2_stage3_dit4sr_testr_5e6_5e5_gtprompt": (
        '/home/cvlab08/projects/data/tair_result/result_val_dit4sr/satext_lv2/fp16_stage3_dit4sr-testr_5e-06-5e-05_lrbranch-attns_ocrloss0.01_descriptive_DiTfeat24_lrstartpoint_gtprompt_/final_restored_img',
        '/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json'
    ),
    "real_benchmark_full_dit4sr_baseline_cfg1_null": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/realtext/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-1__null-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_full_dit4sr_baseline_cfg1_vlm": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/realtext/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-1__pred_vlm-prompt__llava13b__ques-0__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_full_dit4sr_baseline_cfg1_gt": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/realtext/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-1__gt-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_full_dit4sr_baseline_cfg8_null": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/realtext/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-8__null-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_full_dit4sr_baseline_cfg8_vlm": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/realtext/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-8__pred_vlm-prompt__llava13b__ques-0__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_full_dit4sr_baseline_cfg8_gt": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/realtext/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-8__gt-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv2_full_dit4sr_baseline_cfg1_null": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv2/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-1__null-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_full_dit4sr_baseline_cfg1_vlm": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv2/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-1__pred_vlm-prompt__llava_13b__ques-0__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_full_dit4sr_baseline_cfg1_gt": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv2/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-1__gt-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_full_dit4sr_baseline_cfg8_null": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv2/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-8__null-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_full_dit4sr_baseline_cfg8_vlm": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv2/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-8__pred_vlm-prompt__llava_13b__ques-0__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_full_dit4sr_baseline_cfg8_gt": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv2/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-8__gt-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    # Samtext lv3
    "samtext_lv3_full_dit4sr_baseline_cfg1_null": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv3/all__dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-1__null-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_full_dit4sr_baseline_cfg1_vlm": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv3/all__dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-1__pred_vlm-prompt__llava13b__ques-0__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_full_dit4sr_baseline_cfg1_gt": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv3/all__dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-1__gt-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_full_dit4sr_baseline_cfg8_null": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv3/all__dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-8__null-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_full_dit4sr_baseline_cfg8_vlm": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv3/all__dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-8__pred_vlm-prompt__llava13b__ques-0__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_full_dit4sr_baseline_cfg8_gt": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv3/all__dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-8__gt-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    # Samtext lv1
    "samtext_lv1_full_dit4sr_baseline_cfg1_null": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv1/all__dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-1__null-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_full_dit4sr_baseline_cfg1_vlm": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv1/all__dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-1__pred_vlm-prompt__llava_13b__ques-0__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_full_dit4sr_baseline_cfg1_gt": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv1/all__dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-1__gt-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_full_dit4sr_baseline_cfg8_null": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv1/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-8__null-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_full_dit4sr_baseline_cfg8_vlm": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv1/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-8__pred_vlm-prompt__llava_13b__ques-0__/final_restored_img",   
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_full_dit4sr_baseline_cfg8_gt": (
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_2nd/full_eval/satext_lv1/dit4sr_baseline__startpoint-noise__alignmethod-adain__cfg-8__gt-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_terediff_pred_tsmprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_terediff/satext_lv1/terediff_pred_tsmprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_terediff_gtprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_terediff/satext_lv1/terediff_gtprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_terediff_nullprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_terediff/satext_lv1/terediff_nullprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_terediff_pred_tsmprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_terediff/satext_lv2/terediff_pred_tsmprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_terediff_gtprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_terediff/satext_lv2/terediff_gtprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_terediff_nullprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_terediff/satext_lv2/terediff_nullprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_terediff_pred_tsmprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_terediff/satext_lv3/terediff_pred_tsmprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_terediff_gtprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_terediff/satext_lv3/terediff_gtprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_terediff_nullprompt": (
        "/home/cvlab08/projects/data/tair_result/result_val_terediff/satext_lv3/terediff_nullprompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "real_benchmark_stage3_dit4sr_testr_jihye":(
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_3rd/full_eval/realtext/all__dit4sr-testr-trained__stage3__dit4sr-testr__1e-05-1e-05__bs-1__gradaccum-64__lrbranch-attns__ocrloss0.01__descriptive__feat-hq-lq__hidden-after__ir-dit4sr-jihye-s1__startpoint-noise__alignmethod-adain__cfg-1__pred_tsm-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage3_dit4sr_testr_q":( 
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_3rd/full_eval/realtext/all__dit4sr-testr-trained__stage3__dit4sr-testr__1e-05-1e-05__bs-1__gradaccum-64__lrbranch-attns__ocrloss0.01__descriptive__feat-hq-lq__hidden-after__ir-dit4sr-q__startpoint-noise__alignmethod-adain__cfg-1__pred_tsm-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage2_testr_q":(
        "/home/cvlab08/projects/data/tair_result/result_val_dit4sr_3rd/full_eval/realtext/all__testr-trained__stage2__testr__1e-05__bs-1__gradaccum-1____ocrloss0.02__descriptive__feat-hq-lq__hidden-after__ir-dit4sr-q__startpoint-noise__alignmethod-adain__cfg-1__pred_tsm-prompt__/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage1_dit4sr_finetune_lqbranch":(
        "/home/cvlab08/projects/data/tair_result/result_val_FINAL_dit4sr/full_eval/realtext/all__fp16__stage1__dit4sr-5e-05__bs-2__gradaccum-32__finetune-lqbranch__startpoint-noise__alignmethod-adain__cfg-1__null-prompt__FINAL__ir-dit4sr-s1-8k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage3_dit4sr_ocrloss_001_hqlq_3k":(
        "/home/cvlab08/projects/data/tair_result/result_val_FINAL_dit4sr/full_eval/realtext/all__fp16__stage3__dit4sr-1e-05__testr-1e-05__ocrloss0.01__extract-hqlq_feat-num-24__bs-1__gradaccum-64__finetune-lqbranch_straight-to-stage3__startpoint-noise__alignmethod-adain__cfg-1__pred_tsm-prompt__FINAL__ckpt-3k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage3_dit4sr_ocrloss_001_hqlq_5k":(
        "/home/cvlab08/projects/data/tair_result/result_val_FINAL_dit4sr/full_eval/realtext/all__fp16__stage3__dit4sr-1e-05__testr-1e-05__ocrloss0.01__extract-hqlq_feat-num-24__bs-1__gradaccum-64__finetune-lqbranch_straight-to-stage3__startpoint-noise__alignmethod-adain__cfg-1__pred_tsm-prompt__FINAL__ckpt-5k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage3_dit4sr_orcloss_00002_hqlq_3k":(
        "/home/cvlab08/projects/data/tair_result/result_val_FINAL_dit4sr/full_eval/realtext/all__fp16__stage3__dit4sr-1e-05__testr-1e-05__ocrloss0.002__extract-hqlq_feat-num-24__bs-1__gradaccum-64__finetune-lqbranch_straight-to-stage3__startpoint-noise__alignmethod-adain__cfg-1__pred_tsm-prompt__FINAL__ckpt-3k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage3_dit4sr_orcloss_00002_lq_3k":(
        "/home/cvlab08/projects/data/tair_result/result_val_FINAL_dit4sr/full_eval/realtext/all__fp16__stage3__dit4sr-1e-05__testr-1e-05__ocrloss0.002__extract-lq_feat-num-24__bs-1__gradaccum-64__finetune-lqbranch_straight-to-stage3__startpoint-noise__alignmethod-adain__cfg-1__pred_tsm-prompt__FINAL__ckpt-3k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage1_dit4sr_jihye_32k":(
        "/home/cvlab08/projects/data/tair_result/result_val_FINAL_dit4sr/full_eval/realtext/all__jihye_stage1__startpoint-noise__alignmethod-adain__cfg-1__null-prompt__FINAL__ir-dit4sr-s1-32k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage1_jihye_stage2_dit4sr_40k_gt_prompt":(
        "/home/cvlab08/projects/data/jinlovespho/cvpr26/DiT4SR/result_val_FINAL_FINAL_dit4sr/full_eval/realtext/all__jihye_stage1__startpoint-noise__alignmethod-adain__cfg-1__gt-prompt__ir-dit4sr-s1-jihye__tsm-s2-40k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage1_jihye_stage2_dit4sr_40k_null_prompt":(
        "/home/cvlab08/projects/data/jinlovespho/cvpr26/DiT4SR/result_val_FINAL_FINAL_dit4sr/full_eval/realtext/all__jihye_stage1__startpoint-noise__alignmethod-adain__cfg-1__null-prompt__ir-dit4sr-s1-jihye__tsm-s2-40k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage1_jihye_stage2_q_dit4sr_40k_tsm_prompt":(
        "/home/cvlab08/projects/data/jinlovespho/cvpr26/DiT4SR/result_val_FINAL_FINAL_dit4sr/full_eval/realtext/all__jihye_stage1__startpoint-noise__alignmethod-adain__cfg-1__pred_tsm-prompt__ir-dit4sr-q__tsm-s2-40k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage1_jihye_stage2_dit4sr_40k_tsm_prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__finetune-tsm__ckpt-40k__ir-dit4sr-jihye__tsm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_cfg-1__pred_tsm-prompt__extended-stage2__finetune-lq__ckpt1k":(
        "/home/cvlab08/projects/data/jinlovespho/cvpr26/DiT4SR/result_val_FINAL_FINAL_dit4sr/full_eval/realtext/all__fp16__stage2__dit4sr-1e-05__testr-1e-05__ocrloss0.01__extract-hqlq_feat-num-24__bs-1__gradaccum-64__finetune-lq-branch__ir-dit4sr-s1-jihye__tsm-s2-40k__startpoint-noise__alignmethod-adain__cfg-1__pred_tsm-prompt__extended-stage2__finetune-lq__ckpt1k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_cfg-1__pred_tsm-prompt__extended-stage2__ft-lq-hq__ckpt1k":(
        "/home/cvlab08/projects/data/jinlovespho/cvpr26/DiT4SR/result_val_FINAL_FINAL_dit4sr/full_eval/realtext/all__fp16__stage2__dit4sr-1e-05__testr-1e-05__ocrloss0.01__extract-hqlq_feat-num-24__bs-1__gradaccum-64__finetune-lq-hq-branch__ir-dit4sr-s1-jihye__tsm-s2-40k__startpoint-noise__alignmethod-adain__cfg-1__pred_tsm-prompt__extended-stage2__ft-lq-hq__ckpt1k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_cfg-1__pred_tsm-prompt__extended-stage3__ft-lq__ckpt1k":(
        "/home/cvlab08/projects/data/jinlovespho/cvpr26/DiT4SR/result_val_FINAL_FINAL_dit4sr/full_eval/realtext/all__fp16__stage3__dit4sr-1e-05__testr-1e-05__ocrloss0.01__extract-hqlq_feat-num-24__bs-1__gradaccum-64__finetune-lq-branch__ir-dit4sr-s1-jihye__tsm-s2-40k__startpoint-noise__alignmethod-adain__cfg-1__pred_tsm-prompt__extended-stage3__ft-lq__ckpt1k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_cfg-1__pred_tsm-prompt__extended-stage3__ft-lq-hq__ckpt1k":(
        "/home/cvlab08/projects/data/jinlovespho/cvpr26/DiT4SR/result_val_FINAL_FINAL_dit4sr/full_eval/realtext/all__fp16__stage3__dit4sr-1e-05__testr-1e-05__ocrloss0.01__extract-hqlq_feat-num-24__bs-1__gradaccum-64__finetune-lq-hq-branch__ir-dit4sr-s1-jihye__tsm-s2-40k__startpoint-noise__alignmethod-adain__cfg-1__pred_tsm-prompt__extended-stage3__ft-lq-hq__ckpt1k/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv2_stage2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__tsm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__tsm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_stage2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__tsm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__tsm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_STAGE1__finetune-lq__ckpt-8k__null-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/STAGE1__finetune-lq__ckpt-8k__null-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_STAGE1__finetune-lq-hq__ckpt-jihye__null-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/STAGE1__finetune-lq-hq__ckpt-jihye__null-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-q__tsm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-q__tsm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__tsm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__tsm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_STAGE1__finetune-lq-hq__ckpt-jihye__null-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/STAGE1__finetune-lq-hq__ckpt-jihye__null-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-q__tsm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-q__tsm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE1__finetune-lq-hq__ckpt-jihye__null-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE1__finetune-lq-hq__ckpt-jihye__null-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-q__tsm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-q__tsm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__null-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__null-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__gt-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__gt-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE2__finetune-tsm-frozen-ir-lq__ckpt2k__tsm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE2__finetune-tsm-frozen-ir-lq__ckpt2k__tsm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE2__finetune-tsm-frozen-ir-lq-hq__ckpt2k__tsm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE2__finetune-tsm-frozen-ir-lq-hq__ckpt2k__tsm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE3__finetune-tsm-frozen-ir-lq__ckpt2k__tsm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE3__finetune-tsm-frozen-ir-lq__ckpt2k__tsm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE3__finetune-tsm-frozen-ir-lq-hq__ckpt2k__tsm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE3__finetune-tsm-frozen-ir-lq-hq__ckpt2k__tsm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "real_benchmark_stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "real_benchmark_stage1__finetune-lq-hq__ckpt-jihye__gt-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage1__finetune-lq-hq__ckpt-jihye__gt-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques0-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques0-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques3-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques3-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques1-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques1-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques2-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques2-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv1_STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__null-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__null-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__null-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/STAGE2__finetune-tsm__ckpt-40k__ir-dit4sr-s1-jihye__null-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques0-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques0-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques1-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques1-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE1__finetune-lq-hq__ckpt-jihye__gt-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE1__finetune-lq-hq__ckpt-jihye__gt-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques0-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques0-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques1-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques1-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques2-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques2-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques3-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/STAGE1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques3-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "real_benchmark_stage2__vlm-qwenvl-7b-correct-5-10-15-20-25-30-35__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__vlm-qwenvl-7b-correct-5-10-15-20-25-30-35__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage2__vlm-qwenvl-7b-correct-5-10-15-20-25-30-35__cfg8":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__vlm-qwenvl-7b-correct-5-10-15-20-25-30-35__cfg8/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_straight-to-stage3__vlm-qwenvl-7b-correct-5-10-15-20-25-30-35__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/straight-to-stage3__vlm-qwenvl-7b-correct-5-10-15-20-25-30-35__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_straight-to-stage3__vlm-qwenvl-7b-correct-5-10-15-20-25-30-35__cfg8":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/straight-to-stage3__vlm-qwenvl-7b-correct-5-10-15-20-25-30-35__cfg8/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv2_stage2__vlm-qwenvl-7b-correct-5-10-15-20-25-30-35__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/stage2__vlm-qwenvl-7b-correct-5-10-15-20-25-30-35__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_stage2__vlm-qwenvl-7b-correct-5-10-15-20-25-30-35__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/stage2__vlm-qwenvl-7b-correct-5-10-15-20-25-30-35__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "real_benchmark_stage2__FINAL-vlm-qwenvl-7b-correct-10__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__FINAL-vlm-qwenvl-7b-correct-10__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage2__FINAL-vlm-qwenvl-7b-correct-10-20__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__FINAL-vlm-qwenvl-7b-correct-10-20__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage2__FINAL-vlm-qwenvl-7b-correct-20__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__FINAL-vlm-qwenvl-7b-correct-20__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage2__FINAL-vlm-qwenvl-7b-correct-20-25-30-35__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__FINAL-vlm-qwenvl-7b-correct-20-25-30-35__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage2__FINAL-vlm-qwenvl-7b-correct-20-30__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__FINAL-vlm-qwenvl-7b-correct-20-30__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage2__FINAL-vlm-qwenvl-7b-correct-30__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__FINAL-vlm-qwenvl-7b-correct-30__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv3_stage2__FINAL-vlm-qwenvl-7b-correct-20-30__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/stage2__FINAL-vlm-qwenvl-7b-correct-20-30__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_stage2__FINAL-vlm-qwenvl-7b-correct-30__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/stage2__FINAL-vlm-qwenvl-7b-correct-30__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "real_benchmark_stage2__FINAL__finetune-ir-lq-tsm-frozen__ckpt2k__vlm-prompt-ques2-qwenvl-7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__FINAL__finetune-ir-lq-tsm-frozen__ckpt2k__vlm-prompt-ques2-qwenvl-7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage2__FINAL__finetune-ir-lq-hq-tsm-frozen__ckpt2k__vlm-prompt-ques2-qwenvl-7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__FINAL__finetune-ir-lq-hq-tsm-frozen__ckpt2k__vlm-prompt-ques2-qwenvl-7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage3__FINAL__finetune-ir-lq-tsm-frozen__ckpt2k__vlm-prompt-ques2-qwenvl-7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage3__FINAL__finetune-ir-lq-tsm-frozen__ckpt2k__vlm-prompt-ques2-qwenvl-7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage3__FINAL__finetune-ir-lq-hq-tsm-frozen__ckpt2k__vlm-prompt-ques2-qwenvl-7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage3__FINAL__finetune-ir-lq-hq-tsm-frozen__ckpt2k__vlm-prompt-ques2-qwenvl-7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_straight-to-stage3__vlm-prompt-ques2-qwenvl-7b_cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/straight-to-stage3__vlm-prompt-ques2-qwenvl-7b_cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_straight-to-stage3__gt-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/straight-to-stage3__gt-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv3_straight-to-stage3__vlm-prompt-ques2-qwenvl-7b_cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/straight-to-stage3__vlm-prompt-ques2-qwenvl-7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_straight-to-stage3__gt-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/straight-to-stage3__gt-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10-20__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10-20__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-25-30-35__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-25-30-35__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-30__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-30__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-30__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-30__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    #lv 2
    "samtext_lv2_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10-20__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10-20__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-25-30-35__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-25-30-35__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-30__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-30__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv2_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-30__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-30__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "real_benchmark_dreamclear":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/dreamclear",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_jpg.json"
    ),
    #lv 1
    "samtext_lv1_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10-20__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10-20__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-25-30-35__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-25-30-35__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-30__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-20-30__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_stage2__FINAL-vlm-ques1-qwenvl-7b-correct-30__cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-30__cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "real_benchmark_stage2__FINAL-vlm-ques2-qwenvl-7b-correct-10__cfg8":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__FINAL-vlm-ques2-qwenvl-7b-correct-10__cfg8/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_stage2__FINAL-vlm-ques2-qwenvl-7b-correct-10__cfg8-added-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/stage2__FINAL-vlm-ques2-qwenvl-7b-correct-10__cfg8-added-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv3_stage2__FINAL-vlm-ques2-qwenvl-7b-correct-10__cfg8":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10__cfg8/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_stage2__FINAL-vlm-ques2-qwenvl-7b-correct-10__cfg8-added-prompt":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/stage2__FINAL-vlm-ques1-qwenvl-7b-correct-10__cfg8-added-prompt/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    # lvl1 
    "samtext_lv1_stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques0-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques0-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques1-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques1-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques2-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques2-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques3-qwen7b":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/stage1__finetune-lq-hq__ckpt-jihye__vlm-prompt-ques3-qwen7b/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "real_benchmark_dit4sr-baseline__vlm-ques2-qwenvl-7b-cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/dit4sr-baseline__vlm-ques2-qwenvl-7b-cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "real_benchmark_dit4sr-baseline__vlm-ques2-qwenvl-7b-cfg8":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/realtext/dit4sr-baseline__vlm-ques2-qwenvl-7b-cfg8/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/real_benchmark_full/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr-baseline__vlm-ques1-qwenvl-7b-cfg1":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/dit4sr-baseline__vlm-ques1-qwenvl-7b-cfg1/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv3_dit4sr-baseline__vlm-ques1-qwenvl-7b-cfg8":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/dit4sr-baseline__vlm-ques1-qwenvl-7b-cfg8/final_restored_img",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_png.json"
    ),
    "samtext_lv1_dreamclear":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv1/dreamclear",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_jpg.json"
    ),
    "samtext_lv2_dreamclear":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv2/dreamclear",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_jpg.json"
    ),
    "samtext_lv3_dreamclear":(
        "/home/cvlab08/projects/data/tair_result/CVPR26_FINAL_BENCHMARK_RESULTS/satext_test_lv3/dreamclear",
        "/home/cvlab08/projects/data2/text_restoration/100K/test/dataset_coco_jpg.json"
    ),
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
