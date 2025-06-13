#!/bin/bash
set -x

# Apply memory configuration
python3 /app/malteseGEC/tf_memory_config.py

## Docker redundancy
## source ~/virtualenvs/t2t/bin/activate

echo "Venv loaded"
configuration_file=$1
source ${configuration_file}
echo "Configuration file loaded"

echo "Copying vocab"
mkdir -p $NO_EDIT_DATA_DIR
cp $VOCAB_PATH $NO_EDIT_DATA_DIR/vocab.$PROBLEM.32768.subwords
echo "Copied vocab to: $NO_EDIT_DATA_DIR/vocab.$PROBLEM.32768.subwords"

echo "Generating data into ${NO_EDIT_DATA_DIR} for problem: ${PROBLEM}"

# Build t2t-datagen command
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
   --data_ratio=$DATA_RATIO \
   --additional_artificial_sentences=$ADDITIONAL_ARTIFICIAL_SENTENCES \
   --additional_wiki_sentences=$ADDITIONAL_WIKI_SENTENCES \
   --additional_data_filtered=$ADDITIONAL_DATA_FILTERED \
   --input_sentence_file=$INPUT_SENTENCE_FILE \
   --target_sentence_file=$TARGET_SENTENCE_FILE"

# Add mode parameter if it exists
if [ ! -z "$MODE" ]; then
    DATAGEN_CMD="$DATAGEN_CMD --mode=$MODE"
fi

echo "Running datagen: $DATAGEN_CMD"
eval $DATAGEN_CMD

# generate edit weights
VOCAB_FILE=${NO_EDIT_DATA_DIR}/vocab.${PROBLEM}.32768.subwords
echo "Using vocabulary file: ${VOCAB_FILE}"

if [ ! -f "${VOCAB_FILE}" ]; then
    echo "ERROR: Vocabulary file not found at ${VOCAB_FILE}"
    echo "Checking directory contents:"
    ls -la ${NO_EDIT_DATA_DIR}/
    exit 1
fi

echo "Generating train files with edit-weights into ${DATA_DIR}"
TRAIN_FILE_COUNT=$(find "${DATA_DIR}" -name "*-train-*" 2>/dev/null | wc -l)

if [ ! -d "${DATA_DIR}" ] || [ "$TRAIN_FILE_COUNT" -eq 0 ]; then
    echo "Edit weight addition needed"
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
   
    echo "Copying vocab and dev/test files"
    cp ${VOCAB_FILE} ${DATA_DIR}/$(basename ${VOCAB_FILE})
    
    # Copy dev and test files
    for file in ${NO_EDIT_DATA_DIR}/*-dev-* ${NO_EDIT_DATA_DIR}/*-test-*; do
        if [ -f "$file" ]; then
            cp "$file" "${DATA_DIR}/$(basename $file)"
        fi
    done
else
    echo "Edit weight files already exist"
fi

# copy model checkpoints to new directory (not to mess the old pretrained)
echo "Copying old model checkpoints from pretrained dir to new training dir"
mkdir -p $TRAIN_DIR

if [ -d "$PRETRAINED_DIR" ]; then
    echo "Copying from $PRETRAINED_DIR to $TRAIN_DIR"
    cp -r $PRETRAINED_DIR/* $TRAIN_DIR
else
    echo "ERROR: Pretrained directory not found: $PRETRAINED_DIR"
    exit 1
fi

# train
echo "Starting finetuning"
t2t-trainer \
  --data_dir=${DATA_DIR} \
  --problem=${PROBLEM} \
  --model=${MODEL} \
  --hparams_set=${MODEL_TYPE} \
  --hparams="input_word_dropout=${INPUT_WORD_DROPOUT_RATE},target_word_dropout=${TARGET_WORD_DROPOUT_RATE},batch_size=${BATCH_SIZE},max_length=${MAX_LEN},learning_rate_warmup_steps=${WARMUP_STEPS},learning_rate_constant=${LEARNING_RATE_CONSTANT},learning_rate_schedule=constant*rsqrt_decay,optimizer=Adafactor" \
  --output_dir=${TRAIN_DIR} \
  --t2t_usr_dir=${PROBLEM_DIR} \
  --worker_gpu=${NUM_GPUS} \
  --train_steps=300000 \
  --keep_checkpoint_every_n_hours=1 \
  --keep_checkpoint_max=100 \
  --schedule=train \
  --save_checkpoints_secs=600