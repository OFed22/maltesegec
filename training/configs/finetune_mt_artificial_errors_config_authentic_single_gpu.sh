#!/bin/bash
TRANSLATE='gpu'
NUM_GPUS=1
INPUT_WORD_DROPOUT_RATE=0.2
TARGET_WORD_DROPOUT_RATE=0.1
EDIT_WEIGHT=3
MODEL=transformer
MODEL_TYPE=transformer_base_single_gpu_with_dropout
EXPERIMENT_ROOT_DIR=/home/fed/malteseGEC/training
PROBLEM=finetune_general_problem
PROBLEM_DIR=$EXPERIMENT_ROOT_DIR/problems/
TRAIN_DIR=$EXPERIMENT_ROOT_DIR/t2t_train
DATA_DIR=$EXPERIMENT_ROOT_DIR/t2t_data
BATCH_SIZE=1024
MAX_LEN=150
WARMUP_STEPS=10000
LEARNING_RATE_CONSTANT=1

# decoding (Tensorboard)
BEAM_SIZE=4
ALPHA=0.6

# artificial data part - matching extended configuration
TOKEN_ERR_PROB="0.15"
TOKEN_ERR_DISTRIBUTION="0.7_0.1_0.1_0.1_0"
CHAR_ERROR_PROB="0.02"
CHAR_ERR_DISTRIBUTION="0.2_0.2_0.2_0.2_0.2_0"
MODE="extended"
LANG="mt"

# Finetuning specific parameters
DATA_RATIO=5  # 5:1 synthetic to authentic ratio
ADDITIONAL_ARTIFICIAL_SENTENCES=200000
ADDITIONAL_WIKI_SENTENCES=0
ADDITIONAL_DATA_FILTERED="False"

# Authentic data paths - Complete authentic data (Qari + Busuttil + Wiki)
INPUT_SENTENCE_FILE="/app/malteseGEC/data/authentic/splits/train/complete_src.txt"
TARGET_SENTENCE_FILE="/app/malteseGEC/data/authentic/splits/train/complete_trg.txt"

# Technical stuff - paths for extended model
VOCAB_PATH=${DATA_DIR}/artificial_errors-$LANG-$MODE-tep$TOKEN_ERR_PROB-ted$TOKEN_ERR_DISTRIBUTION-cep$CHAR_ERROR_PROB-ced$CHAR_ERR_DISTRIBUTION/vocab.artificial_errors.32768.subwords
NO_EDIT_DATA_DIR=${DATA_DIR}/$PROBLEM-$LANG-complete-finetune-tep$TOKEN_ERR_PROB-ted$TOKEN_ERR_DISTRIBUTION-cep$CHAR_ERROR_PROB-ced$CHAR_ERR_DISTRIBUTION-aas${ADDITIONAL_ARTIFICIAL_SENTENCES}-dr${DATA_RATIO}
DATA_DIR_WITH_WEIGHT=${NO_EDIT_DATA_DIR}-${EDIT_WEIGHT}

# Pretrained model directory (extended model)
PRETRAINED_DIR=${TRAIN_DIR}/artificial_errors-$LANG-$MODE-tep$TOKEN_ERR_PROB-ted$TOKEN_ERR_DISTRIBUTION-cep$CHAR_ERROR_PROB-ced$CHAR_ERR_DISTRIBUTION/$MODEL-$MODEL_TYPE-iwdr${INPUT_WORD_DROPOUT_RATE}-twdr${TARGET_WORD_DROPOUT_RATE}-ew${EDIT_WEIGHT}-ws10000-lrc1

# New training directory for finetuned model
TRAIN_DIR_FINETUNED=${TRAIN_DIR}/$PROBLEM-$LANG-complete-finetune-tep$TOKEN_ERR_PROB-ted$TOKEN_ERR_DISTRIBUTION-cep$CHAR_ERROR_PROB-ced$CHAR_ERR_DISTRIBUTION/$MODEL-$MODEL_TYPE-iwdr${INPUT_WORD_DROPOUT_RATE}-twdr${TARGET_WORD_DROPOUT_RATE}-ew${EDIT_WEIGHT}-ws${WARMUP_STEPS}-lrc${LEARNING_RATE_CONSTANT}-aas${ADDITIONAL_ARTIFICIAL_SENTENCES}-dr${DATA_RATIO}

# Override for compatibility with script
DATA_DIR=${DATA_DIR_WITH_WEIGHT}
TRAIN_DIR=${TRAIN_DIR_FINETUNED}

# Memory management settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_DISABLE_CUBLAS_GEMM=1
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=0
export TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=0
export NVIDIA_TF32_OVERRIDE=0
export CUDA_CACHE_DISABLE=1
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1