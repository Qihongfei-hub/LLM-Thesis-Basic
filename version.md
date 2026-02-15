# Version Information

## Version-main
- **Version**: 6.1.1 on main 
- **Release Date**: 2026-02-15


## Version-main
- **Version**: 5.1.1 on main 
- **Release Date**: 2026-02-14


## Version-main
- **Version**: 4.1.1 on Trunk
- **Release Date**: 2026-02-09


## Version-main
- **Version**: 5.1.1 on Trunk
- **Release Date**: 2026-02-14


## Version-main
- **Version**: 3.1.1 on Trunk
- **Release Date**: 2026-02-08

## Version-Trunk
- **Version**: 2.1.1 on Trunk
- **Release Date**: 2026-02-08

## History Version-3 -trunk
- **Version**: 1.1.1 on Trunk
- **Release Date**: 2026-02-08

## History Version-2-main
- **Version**: 1.0.1
- **Release Date**: 2026-02-08


## History Version-1-main
- **Version**: 1.0.0
- **Release Date**: 2026-02-08

## Change Log
### v1.0.0 (2026-02-08)
- Initial release



!swift sft --model_id_or_path ./LLM-Research/Meta-Llama-3___1-8B-Instruct \
    --model_type llama3_1-8b-instruct \
    --sft_type lora \
    --tuner_backend peft \
    --template_type AUTO \
    --dtype AUTO \
    --output_dir output \
    --dataset ./ruozhiba-llama3/ruozhiba_qa.json \
    --train_dataset_sample -1 \
    --num_train_epochs 5 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 