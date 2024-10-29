#!/bin/bash

# List of configuration values to iterate over
CONFIG_PATHS=(
    "/raid/speech/soumenmondal/LangSpecificNeurons/outputs/ckpt/Meta-Llama-3.1-8B_finetune_XNLI/act_prob_zero_data_en_frozen_ur_0.25_1.0e-05_r8/master_config.pkl"
)

CKPT_NAMES=(
    "checkpoint-12268/pytorch_model.bin"
)

EVAL_LANGS=(
    "ur"
)

IS_ZERO_SHOTS=(
    1
)

BATCH_SIZE=8
EVAL_FRAC=1.0

# Loop over different values
for i in "${!CONFIG_PATHS[@]}"; do
    export CONFIG_PATH=${CONFIG_PATHS[$i]}
    export CKPT_NAME=${CKPT_NAMES[$i]}
    export EVAL_LANG=${EVAL_LANGS[$i]}
    export BATCH_SIZE=${BATCH_SIZE}
    export EVAL_FRAC=${EVAL_FRAC}
    export IS_ZERO_SHOT=${IS_ZERO_SHOTS[$i]}
    
    echo "Running evaluation with:"
    echo "Config Path: $CONFIG_PATH"
    echo "Checkpoint Name: $CKPT_NAME"
    echo "Eval Lang: $EVAL_LANG"
    echo "Is zero shot: $IS_ZERO_SHOT"
    
    nohup python src/task_eval.py > task_eval1_${i}.out & 
done
