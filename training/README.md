# Training Directory

## Structure

```
training/
├── configs/                    # Configuration files for experiments
├── problems/                   # T2T problem definitions
│   ├── artificial_errors.py    # Pretraining problem
│   └── finetune_general_problem.py  # Finetuning problem
├── run_training_memory.sh      # Pretraining script
├── run_finetuning_memory.sh    # Finetuning script
├── t2t_data/                   # Generated TFRecord data (created during training)
└── t2t_train/                  # Model checkpoints (created during training)
```

## Pretraining

Run pretraining experiments using Docker:

```bash
docker run -it --rm --gpus all \
  -v ~/malteseGEC:/app/malteseGEC \
  -v ~/malteseGEC:/home/fed/malteseGEC \
  maltesegec-ngc-fixed \
  bash -c 'cd /app/malteseGEC && \
      cp cpu_patch_trainer.py /usr/local/bin/t2t-trainer && \
      ./training/run_training_memory.sh ./training/configs/mt_artificial_errors_config_base_single_gpu_memory.sh'
```

Available pretraining configs:
- `mt_artificial_errors_config_base_single_gpu_memory.sh`
- `mt_artificial_errors_config_extended_single_gpu_memory.sh`
- `mt_artificial_errors_config_asr_single_gpu_memory.sh`
- `mt_artificial_errors_config_debattista_single_gpu_memory.sh`
- `mt_artificial_errors_config_busuttil_single_gpu_memory.sh`

## Finetuning

Run finetuning experiments using Docker:

```bash
docker run -it --rm --gpus all \
  -v ~/malteseGEC:/app/malteseGEC \
  -v ~/malteseGEC:/home/fed/malteseGEC \
  maltesegec-ngc-fixed \
  bash -c 'cd /app/malteseGEC && \
      cp cpu_patch_trainer.py /usr/local/bin/t2t-trainer && \
      ./training/run_finetuning_memory.sh ./training/configs/finetune_mt_artificial_errors_config_qari_single_gpu.sh'
```

Available finetuning configs:
- `finetune_mt_artificial_errors_config_qari_single_gpu.sh`
- `finetune_mt_artificial_errors_config_busuttil_single_gpu.sh`
- `finetune_mt_artificial_errors_config_wiki_single_gpu.sh`
- `finetune_mt_artificial_errors_config_authentic_single_gpu.sh`