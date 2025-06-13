# Data Directory

## Structure

```
data/
├── authentic/          # Human-annotated error corrections
│   ├── raw/           # Original datasets (Debattista, Busuttil)
│   ├── splits/        # Train/dev/test splits
│   └── wikipedia/     # Extracted Wikipedia revisions
├── mixed/             # Pre-existing datasets for pretraining
│   ├── debattista/    # 11.5K mixed authentic+synthetic
│   └── busuttil/      # 517K mostly synthetic
├── monolingual/       # Clean Maltese text sources
│   ├── korpus_malti/  # 20M+ sentences
│   └── common_voice/  # Validated transcriptions
├── synthetic/         # Combined monolingual sources
├── chunks/            # Generated synthetic error data
│   └── mt/           # Maltese chunks by mode
└── vocabularies/      # Subword vocabularies
```

## Dataset preperation

Some datasets could not be uploaded to the repository as a whole, if you wish to use them use

```
cat data/synthetic/basic_combined_monolingual_part_* > data/synthetic/basic_combined_monolingual.txt
cat data/synthetic/extended_combined_monolingual_part_* > data/synthetic/extended_combined_monolingual.txt
cat data/monolingual/korpus_malti/processed/sentences_part_* > data/monolingual/korpus_malti/processed/sentences.txt

## Data Generation

Generate synthetic error data using Docker:

```bash
# Example: Generate basic mode chunks (3 parallel jobs)
docker run -it --rm --gpus all \
  -v ~/malteseGEC:/app/malteseGEC \
  -v ~/malteseGEC:/home/fed/malteseGEC \
  -w /app/malteseGEC/data \
  maltesegec-ngc-fixed \
  bash -c 'generate_data_wrapper.sh basic 3'
```

Available modes: `basic`, `extended`, `asr`, `debattista`, `busuttil`

## Key Files

- `generate_data_wrapper.sh` - Main data generation script
- `introduce_errors.py` - Error synthesis implementation