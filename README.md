# Maltese Grammatical Error Correction

## Docker Setup

Build the Docker image for training, evaluation, or data processing:
```bash
docker build -f Dockerfile.ngc -t maltesegec-ngc-fixed .
```

## Directory Structure

```
malteseGEC/
├── data/               # All datasets and data generation scripts
├── training/           # Training scripts and configurations
├── eval/         # Evaluation scripts and metrics
├── maltese_wiki/       # Wikipedia extraction scripts
```
## Requirements

All requirements are handled by the docker, which aims to replicate the environment used by Náplava and Straka's original work. Additionally, the docker contains some environment variables to enable compatability with modern GPUs.
cpu_patch_trainer.py serves as a patch to some problematic CUBLAS errors encountered during the project's development, enabling a cpu fallback for operations incompatible with modern GPUs.