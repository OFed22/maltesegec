source ~/virtualenvs/aspell/bin/activate # virtualenv with all requirements (mainly aspell-python-py3)

set -ex

chunk_number=$1
max_chunks=$2
mode=$3  # basic, extended, or asr

token_err_prob=$4
token_err_distribution=$5

char_err_prob=$6
char_err_distribution=$7

lang=$8
monolingual_data=$9
vocabulary=${10}

# Additional parameters for extended and ASR modes
extended_token_distribution=${11:-"0.65_0.05_0.05_0.05_0.1_0.1"}
reverse_prob=${12:-"0.02"}
mono_prob=${13:-"0.02"}
asr_mapping_file=${14:-""}
asr_prob=${15:-"0.04"}

dirname=chunks/$lang/$mode/$token_err_prob-$token_err_distribution-$char_err_prob-$char_err_distribution
mkdir -p "$dirname"

python_cmd="python3 introduce_errors.py $vocabulary --lang=$lang --mode=$mode --token_err_prob=$token_err_prob --token_err_distribution=$token_err_distribution --char_err_prob=$char_err_prob --char_err_distribution=$char_err_distribution"

# Add mode-specific parameters
if [ "$mode" = "extended" ]; then
    python_cmd="$python_cmd --extended_token_distribution=$extended_token_distribution --reverse_prob=$reverse_prob --mono_prob=$mono_prob"
elif [ "$mode" = "asr" ]; then
    if [ -n "$asr_mapping_file" ] && [ -f "$asr_mapping_file" ]; then
        python_cmd="$python_cmd --asr_mapping_file=$asr_mapping_file --asr_prob=$asr_prob"
    fi
fi

split --number=l/$chunk_number/$max_chunks $monolingual_data | eval $python_cmd > "$dirname/chunk_$chunk_number-$max_chunks.txt"

echo "Generated chunk $chunk_number/$max_chunks for mode $mode"