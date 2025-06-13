# RNALens

## Install

1. Create a conda environment.
```
conda create -n rnalens python==3.9.13
conda activate rnalens
```
2. In RNALens environment, install the `rnalens` library and the Python packages specified in the `pyproject.toml` file.
```bash
pip install -e .
```

## Download Pretrained Model 

The pretrained model is available at [Huggingface](https://huggingface.co/oomics/RNALens). You can download it and fune-tune the model on downstream tasks.

## Data

For downstream tasks, you can find the data [here](https://github.com/a96123155/UTR-LM?tab=readme-ov-file#file-structure). Specifically, we use the following three datasets:
- [HEK_sequence](https://codeocean.com/capsule/4214075/tree/v1/data/TE_REL_Endogenous_Cao/HEK_sequence.csv)
- [Muscle_sequence](https://codeocean.com/capsule/4214075/tree/v1/data/TE_REL_Endogenous_Cao/Muscle_sequence.csv)
- [pc3_sequence](https://codeocean.com/capsule/4214075/tree/v1/data/TE_REL_Endogenous_Cao/pc3_sequence.csv)

## Finetune

To finetune the model, run the following command:

```bash
cd scripts/train

for data in HEK pc3 Muscle
do 
    python \
        run_finetune.py \
        --data_fpath ../../data/cell_line/${data}_sequence.csv \
        --tokenizer_name_or_path <tokenizer_name> \
        --model_name_or_path /path/to/pretrained/model \
        --seq_type utr \
        --label_type rnaseq_log \
        --learning_rate 5e-5 \
        --output-dir ../pretrain-outs/${data} \
        --num_train_epochs 300 \
        --save_total_limit 4 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --dataloader_num_workers 4 \
        --dataloader_prefetch_factor 2 \
        --use_liger_kernel \
        --torch_compile \
        --eval_strategy epoch \
        --save_strategy epoch \
        --load_best_model_at_end \
        --metric_for_best_model Spearman \
        --greater_is_better True \
        --no-save-safetensors \
        --warmup_steps 500
done
```
