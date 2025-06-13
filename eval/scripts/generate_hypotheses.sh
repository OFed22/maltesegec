#!/bin/bash

set -e

# Configuration
BASE_DIR="/home/fed/malteseGEC"
EVAL_DIR="$BASE_DIR/eval"
MODEL_NAME=$1
CHECKPOINT_STEP=$2

if [ -z "$MODEL_NAME" ] || [ -z "$CHECKPOINT_STEP" ]; then
    echo "Usage: $0 <model_name> <checkpoint_step>"
    echo ""
    echo "Available models:"
    echo "  basic     - Basic pretrained model"
    echo "  extended  - Extended pretrained model"
    echo "  finetuned - Extended + finetuning"
    echo ""
    echo "Example: $0 extended 150000"
    exit 1
fi

# Set model-specific paths
case $MODEL_NAME in
    "basic")
        MODEL_DIR="$BASE_DIR/training/t2t_train/artificial_errors-mt-tep0.15-ted0.7_0.1_0.1_0.1-cep0.02-ced0.2_0.2_0.2_0.2_0.2/transformer-transformer_base_single_gpu_with_dropout-iwdr0.2-twdr0.1-ew3-ws10000-lrc1"
        DATA_DIR="$BASE_DIR/training/t2t_data/artificial_errors-mt-tep0.15-ted0.7_0.1_0.1_0.1-cep0.02-ced0.2_0.2_0.2_0.2_0.2-3"
        PROBLEM="artificial_errors"
        ;;
    "extended")
        MODEL_DIR="$BASE_DIR/training/t2t_train/artificial_errors-mt-extended-tep0.15-ted0.7_0.1_0.1_0.1_0-cep0.02-ced0.2_0.2_0.2_0.2_0.2_0/transformer-transformer_base_single_gpu_with_dropout-iwdr0.2-twdr0.1-ew3-ws10000-lrc1"
        DATA_DIR="$BASE_DIR/training/t2t_data/artificial_errors-mt-extended-tep0.15-ted0.7_0.1_0.1_0.1_0-cep0.02-ced0.2_0.2_0.2_0.2_0.2_0-3"
        PROBLEM="artificial_errors"
        ;;
    "finetuned")
        MODEL_DIR="$BASE_DIR/training/t2t_train/finetune_general_problem-mt-extended-finetune-tep0.15-ted0.7_0.1_0.1_0.1_0-cep0.02-ced0.2_0.2_0.2_0.2_0.2_0/transformer-transformer_base_single_gpu_with_dropout-iwdr0.2-twdr0.1-ew3-ws10000-lrc1-aas200000-dr2"
        DATA_DIR="$BASE_DIR/training/t2t_data/finetune_general_problem-mt-extended-finetune-tep0.15-ted0.7_0.1_0.1_0.1_0-cep0.02-ced0.2_0.2_0.2_0.2_0.2_0-aas200000-dr2-3"
        PROBLEM="finetune_general_problem"
        ;;
    *)
        echo "Error: Unknown model '$MODEL_NAME'"
        exit 1
        ;;
esac

OUTPUT_DIR="$EVAL_DIR/results/${MODEL_NAME}_step${CHECKPOINT_STEP}"
mkdir -p $OUTPUT_DIR

CHECKPOINT_PATH="$MODEL_DIR/model.ckpt-${CHECKPOINT_STEP}"
if [ ! -f "${CHECKPOINT_PATH}.index" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Available checkpoints in $MODEL_DIR:"
    ls -la $MODEL_DIR/model.ckpt-*.index 2>/dev/null | sed 's/\.index$//' | tail -10
    exit 1
fi

echo "========================================"
echo "Generating Hypotheses"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Checkpoint: $CHECKPOINT_STEP"
echo "Output directory: $OUTPUT_DIR"
echo ""

RAW_PREDICTIONS="$OUTPUT_DIR/predictions_raw.txt"
PREDICTIONS_FILE="$OUTPUT_DIR/predictions.txt"

if [ -f "$PREDICTIONS_FILE" ]; then
    echo "Predictions already exist at $PREDICTIONS_FILE"
    echo "Delete this file to regenerate predictions."
    exit 0
fi

echo "Step 1: Generating predictions with t2t-decoder..."

# Apply CPU patch
if [ -f "$EVAL_DIR/scripts/cpu_patch_decoder.py" ]; then
    cp $EVAL_DIR/scripts/cpu_patch_decoder.py /usr/local/bin/t2t-decoder
    echo "Applied CPU patch for decoder"
elif [ -f "$BASE_DIR/cpu_patch_trainer.py" ]; then
    cp $BASE_DIR/cpu_patch_trainer.py /usr/local/bin/t2t-decoder
    echo "Applied CPU patch (trainer version) for decoder"
fi

t2t-decoder \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=transformer \
    --hparams_set=transformer_base_single_gpu_with_dropout \
    --output_dir=$MODEL_DIR \
    --checkpoint_path=$CHECKPOINT_PATH \
    --decode_hparams="beam_size=4,alpha=0.6" \
    --decode_from_file=$BASE_DIR/data/authentic/splits/test/src.txt \
    --decode_to_file=$RAW_PREDICTIONS \
    --t2t_usr_dir=$BASE_DIR/training/problems 2>&1 | tee $OUTPUT_DIR/decode.log

if [ ! -f "$RAW_PREDICTIONS" ]; then
    echo "Error: Decoding failed. Check $OUTPUT_DIR/decode.log for details."
    exit 1
fi

echo ""
echo "Step 2: Cleaning predictions..."

sed 's/@@ //g' $RAW_PREDICTIONS | \
sed 's/@@//g' | \
sed 's/<unk>//gi' | \
sed 's/<UNK>//g' | \
sed 's/  */ /g' | \
sed 's/^ *//;s/ *$//' > $PREDICTIONS_FILE

NUM_PREDICTIONS=$(wc -l < $PREDICTIONS_FILE)
NUM_SOURCE=$(wc -l < $BASE_DIR/data/authentic/splits/test/src.txt)

echo ""
echo "Step 3: Verification"
echo "Source sentences: $NUM_SOURCE"
echo "Predictions generated: $NUM_PREDICTIONS"

if [ "$NUM_PREDICTIONS" -ne "$NUM_SOURCE" ]; then
    echo "Warning: Number of predictions doesn't match source!"
fi

echo ""
echo "========================================"
echo "Hypothesis generation complete!"
echo "Raw predictions: $RAW_PREDICTIONS"
echo "Clean predictions: $PREDICTIONS_FILE"
echo ""
echo "Sample predictions (first 3):"
echo "----------------------------------------"
head -3 $PREDICTIONS_FILE
echo "----------------------------------------"