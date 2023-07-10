# TOAST for Language Generation

<img src="assets/vicuna.png" alt="drawing" width="800"/>

## Environment settings

```bash
pip install -r requirements.txt
```

## Model Zoo

|                         Name                          |                                        Weights                                         |
|:-----------------------------------------------------:|:--------------------------------------------------------------------------------------:|
|      LLaMA-7B-topdown (pre-tuned on OpenWebText)      | [weights](https://berkeley.box.com/shared/static/j4ulxk1cr56wih6lpu81x95uti65loqk.bin) |
| Vicuna-7B-TOAST (tuned on ShareGPT data using TOAST)  | [weights](https://berkeley.box.com/shared/static/kgq7i1xlq3ab97ulg6jdve8s8nv4jrxv.bin) |
| Vicuna-13B-TOAST (tuned on ShareGPT data using TOAST) | [weights](https://berkeley.box.com/shared/static/y0xiwpl46fgtwbw22ks5nxdtgcklmpa9.bin) |


## Running

### Inference

To test Vicuna-7B-TOAST on custom instructions, please first download the tuned TOAST weights and then run
```
python inference.py --model_name_or_path decapoda-research/llama-7b-hf --bf16 True --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' --tf32 True --report_to "none" --output_dir <output_path> --model "llama-topdown" --checkpoint <path_to_tuned_topdown_weights>
```

We have some preset questions in `inference.py`. Feel free to change them to your own instructions!

### Tuning Vicuna-7B-TOAST on ShareGPT Data

To train TOAST on ShareGPT data, first download the data from [here](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered).

Then download the pretuned top-down weights and then run
```
torchrun --nproc_per_node=4 --master_port=5959 train.py --model_name_or_path decapoda-research/llama-7b-hf --data_path <path_to_data_folder/ShareGPT_V3_unfiltered_cleaned_split.json> --bf16 True --output_dir <output_path> --num_train_epochs 2 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 12 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --report_to "none" --deepspeed "./configs/default_offload_opt_param.json" --model "llama-topdown" --checkpoint <path_to_pretuned_weight>
```

To try fully fine-tuning or LoRA, please set `--model "llama"` or `--model "llama-lora"` and discard the `--checkpoint` argument.

If you would like to test your own tuned model, you need first convert the format of saved model checkpoints. To do this, go to the output path and run
```
python zero_to_fp32.py . tuned_weights.bin
```

This will save your model checkpoint into `tuned_weights.bin`, which is ready to use for inference.

### Pre-tuning on OpenWebText

To pre-tune the top-down module, run
```
torchrun --nproc_per_node=4 --master_port=4455 pretune_top_down.py --model_name_or_path decapoda-research/llama-7b-hf --data_path ./alpaca_data.json --bf16 True --output_dir <output_path> --num_train_epochs 1 --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --gradient_accumulation_steps 12 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 3e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --report_to "none" --deepspeed "./configs/default_offload_opt_param.json"
```

You also need to run `zero_to_fp32.py` to convert the saved checkpoint for further tuning.