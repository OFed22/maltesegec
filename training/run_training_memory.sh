#!/bin/bash
set -x

# Apply memory configuration
python3 /app/malteseGEC/tf_memory_config.py

## Docker redundancy
## source ~/virtualenvs/t2t/bin/activate

configuration_file=$1
source ${configuration_file}

echo "Configuration file loaded"
echo "Creating data directory ${NO_EDIT_DATA_DIR}"
mkdir -p ${NO_EDIT_DATA_DIR}
echo "Generating data into ${NO_EDIT_DATA_DIR} for problem: ${PROBLEM}"

# Build t2t-datagen command with appropriate parameters
DATAGEN_CMD="t2t-datagen \
   --t2t_usr_dir=\"${PROBLEM_DIR}\" \
   --data_dir=\"${NO_EDIT_DATA_DIR}\" \
   --tmp_dir=/tmp/${PROBLEM} \
   --problem=$PROBLEM \
   --token_err_prob=\"$TOKEN_ERR_PROB\" \
   --token_err_distribution=\"$TOKEN_ERR_DISTRIBUTION\" \
   --char_err_prob=\"$CHAR_ERROR_PROB\" \
   --char_err_distribution=\"$CHAR_ERR_DISTRIBUTION\" \
   --lang=$LANG \
   --mode=$MODE"

# Add extended parameters if mode is extended
if [ "$MODE" = "extended" ]; then
    DATAGEN_CMD="$DATAGEN_CMD \
        --extended_token_distribution=\"$EXTENDED_TOKEN_DISTRIBUTION\" \
        --reverse_prob=\"$REVERSE_PROB\" \
        --mono_prob=\"$MONO_PROB\""
fi

echo "Running datagen command: $DATAGEN_CMD"
eval $DATAGEN_CMD

# generate edit weights
VOCAB_FILE=${NO_EDIT_DATA_DIR}/vocab.artificial_errors.32768.subwords

echo "Generating train files with edit-weights into ${DATA_DIR}"
TRAIN_FILE_COUNT=$(find "${DATA_DIR}" -name "*-train-*" 2>/dev/null | wc -l)

if [ ! -d "${DATA_DIR}" ] || [ "$TRAIN_FILE_COUNT" -eq 0 ]; then
    echo "Edit weight addition needed (directory missing or no training files found)"
    mkdir -p ${DATA_DIR}
    
    for tf_record in ${NO_EDIT_DATA_DIR}/*-train-*; do
        echo "Processing $(basename $tf_record) with edit weight ${EDIT_WEIGHT}"
        python3 /app/malteseGEC/training/add_weights_to_tfrecord.py \
            "${tf_record}" \
            "${DATA_DIR}" \
            "${VOCAB_FILE}" \
            "${EDIT_WEIGHT}"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to process $tf_record"
            exit 1
        fi
    done
   
    echo "Copying vocab file to edit-weighted directory"
    cp "${VOCAB_FILE}" "${DATA_DIR}/$(basename ${VOCAB_FILE})"

    echo "Copying dev and test files to edit-weighted directory"
    for file in ${NO_EDIT_DATA_DIR}/*-dev-* ${NO_EDIT_DATA_DIR}/*-test-*; do
        if [ -f "$file" ]; then
            cp "$file" "${DATA_DIR}/$(basename $file)"
        fi
    done

    
    echo "Edit weight processing complete. Files in ${DATA_DIR}:"
    ls -la "${DATA_DIR}/" | wc -l
else
    echo "Edit weight addition already completed (found $TRAIN_FILE_COUNT training files)"
fi

# train
echo "Training"
t2t-trainer \
  --data_dir=${DATA_DIR} \
  --problem=${PROBLEM} \
  --model=${MODEL} \
  --hparams_set=${MODEL_TYPE} \
  --hparams="input_word_dropout=${INPUT_WORD_DROPOUT_RATE},target_word_dropout=${TARGET_WORD_DROPOUT_RATE},batch_size=${BATCH_SIZE},max_length=${MAX_LEN},learning_rate_warmup_steps=${WARMUP_STEPS},learning_rate_constant=${LEARNING_RATE_CONSTANT},learning_rate_schedule=constant*rsqrt_decay,optimizer=Adafactor" \
  --output_dir=${TRAIN_DIR} \
  --t2t_usr_dir=${PROBLEM_DIR} \
  --worker_gpu=${NUM_GPUS} \
  --train_steps=150000 \
  --keep_checkpoint_every_n_hours=2 \
  --keep_checkpoint_max=100 \
  --schedule=continuous_train_and_eval \
  --eval_steps=1000 \
  --save_checkpoints_secs=3600 \
  --eval_throttle_seconds=3600