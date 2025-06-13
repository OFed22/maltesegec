set -e

lang="mt"

config=${1:-"basic"}  # basic, extended, asr, debattista, busuttil
max_chunks=${2:-3}    # Number of parallel jobs

BASE_DIR="$(pwd)"
DATA_DIR="$BASE_DIR/../data"
KORPUS_DATA="$DATA_DIR/monolingual/korpus_malti/processed/sentences.txt"
CV_DATA="$DATA_DIR/monolingual/common_voice/processed/sentences.txt"
VOCABULARY="$DATA_DIR/vocabularies/vocabulary_mt.tsv"

# ASR-specific files
ASR_MAPPING_FILE="$DATA_DIR/monolingual/common_voice/processed/asr_mappings.tsv"
CV_PATH_MAPPING="$DATA_DIR/monolingual/common_voice/processed/path_mapping.tsv"

# Create vocabulary file if it doesn't exist
if [ ! -f "$VOCABULARY" ]; then
    echo "Creating vocabulary file..."
    mkdir -p "$(dirname "$VOCABULARY")"
    
    # Extract vocabulary from monolingual data
    if [ -f "$KORPUS_DATA" ]; then
        cat "$KORPUS_DATA" | tr ' ' '\n' | grep -E '^[a-zA-ZĊċĠġĦħŻż]+$' | sort | uniq -c | sort -nr | head -50000 | awk '{print $2"\t"$1}' > "$VOCABULARY"
        echo "Created vocabulary with $(wc -l < "$VOCABULARY") entries"
    else
        echo "Error: Korpus Malti data not found at $KORPUS_DATA"
        exit 1
    fi
fi

# Configuration-specific parameters
case $config in
    "basic")
        echo "Using BASIC synthetic dataset configuration"
        mode="basic"
        token_err_prob="0.15"
        token_err_distribution="0.7_0.1_0.1_0.1_0"  # replace_insert_delete_swap_recase
        char_err_prob="0.02" 
        char_err_distribution="0.2_0.2_0.2_0.2_0.2_0"  # replace_insert_delete_swap_change_diacritics_anglicize
        monolingual_data="$KORPUS_DATA"
        
        # Combine with Common Voice
        if [ -f "$CV_DATA" ]; then
            combined_data="$DATA_DIR/synthetic/basic_combined_monolingual.txt"
			mkdir -p "$(dirname "$combined_data")"
            cat "$KORPUS_DATA" "$CV_DATA" > "$combined_data"
            monolingual_data="$combined_data"
            echo "Combined Korpus Malti ($(wc -l < "$KORPUS_DATA") lines) and Common Voice ($(wc -l < "$CV_DATA") lines)"
        fi
        ;;
        
    "extended")
        echo "Using EXTENDED synthetic dataset configuration"
        mode="extended"
        token_err_prob="0.15"
        token_err_distribution="0.7_0.1_0.1_0.1_0"  # Basic distribution for compatibility
        extended_token_distribution="0.65_0.05_0.05_0.05_0.1_0.1"  # replace_insert_delete_swap_unk_replace
        char_err_prob="0.02"
        char_err_distribution="0.2_0.2_0.2_0.2_0.2_0"
        reverse_prob="0.02"
        mono_prob="0.02"
        
        # Combine monolingual sources
        combined_data="$DATA_DIR/synthetic/extended_combined_monolingual.txt"
        cat "$KORPUS_DATA" "$CV_DATA" > "$combined_data"
        monolingual_data="$combined_data"
        ;;
        
    "asr")
        echo "Using ASR-INTEGRATED synthetic dataset configuration"
        mode="asr"
        token_err_prob="0.15"
        token_err_distribution="0.7_0.1_0.1_0.1_0"
        char_err_prob="0.02"
        char_err_distribution="0.2_0.2_0.2_0.2_0.2_0"
        asr_prob="0.04"
        
        # Create ASR mapping if it doesn't exist
        if [ ! -f "$ASR_MAPPING_FILE" ]; then
            echo "Warning: ASR mapping file not found at $ASR_MAPPING_FILE"
            echo "Creating placeholder ASR mapping..."
            mkdir -p "$(dirname "$ASR_MAPPING_FILE")"
            # Create empty mapping for now - you'll need to populate this with actual ASR transcriptions
            touch "$ASR_MAPPING_FILE"
        fi
        
        # Use basic combined data
        combined_data="$DATA_DIR/synthetic/asr_combined_monolingual.txt"
        cat "$KORPUS_DATA" "$CV_DATA" > "$combined_data"
        monolingual_data="$combined_data"
        ;;
        
    "debattista"|"busuttil")
        echo "Using ${config^^} dataset configuration"
        if [ "$config" = "debattista" ]; then
            data_file="$DATA_DIR/mixed/debattista/combined.txt"
        else
            data_file="$DATA_DIR/mixed/busuttil/combined.txt"
        fi
        
        chunks_dir="$DATA_DIR/chunks/$lang/$mode"
        mkdir -p "$chunks_dir"
        ln -sf "$data_file" "$chunks_dir/preexisting_data.txt"
        
        echo "Dataset linked: $(wc -l < "$data_file") lines"
        exit 0
        ;;
            
        *)
            echo "Error: Unknown configuration '$config'"
            echo "Available configurations: basic, extended, asr, debattista, busuttil"
            exit 1
            ;;
    esac

if [ ! -f "$monolingual_data" ]; then
    echo "Error: Monolingual data not found at $monolingual_data"
    exit 1
fi

if [ ! -f "$VOCABULARY" ]; then
    echo "Error: Vocabulary file not found at $VOCABULARY"
    exit 1
fi

echo "Input data: $monolingual_data ($(wc -l < "$monolingual_data") lines)"
echo "Vocabulary: $VOCABULARY ($(wc -l < "$VOCABULARY") entries)"
echo "Mode: $mode"
echo "Token error prob: $token_err_prob"
echo "Char error prob: $char_err_prob"

chunks_dir="chunks/$lang/$mode"
mkdir -p "$chunks_dir"

echo "Starting $max_chunks parallel jobs..."

job_pids=()
for i in $(seq 1 $max_chunks); do
    echo "Starting chunk $i/$max_chunks..."
    
    # Build command arguments
    args=("$i" "$max_chunks" "$mode" "$token_err_prob" "$token_err_distribution" "$char_err_prob" "$char_err_distribution" "$lang" "$monolingual_data" "$VOCABULARY")
    
    # Add mode-specific arguments
    if [ "$mode" = "extended" ]; then
        args+=("$extended_token_distribution" "$reverse_prob" "$mono_prob")
    elif [ "$mode" = "asr" ]; then
        args+=("" "" "" "$ASR_MAPPING_FILE" "$asr_prob")
    fi
    
    bash generate_data.sh "${args[@]}" &
    job_pids+=($!)
done

echo "Waiting for all chunks to complete..."
for pid in "${job_pids[@]}"; do
    wait $pid
    echo "Chunk with PID $pid completed"
done