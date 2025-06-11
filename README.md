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


## Finetune

To finetune the model, run the following command:

```bash
cd scripts/train

for data in HEK pc3 Muscle
do 
    python \
        run_pretrain.py \
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
