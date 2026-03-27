# Lab 4 Pre-lab Checklist (LoRA Finetuning on HPC4)

Course: IEDA 4000I  
Lab: Lab 4 - LoRA/QLoRA Finetuning on HPC4

Complete this checklist before class. The goal is to avoid setup/debug delays during GPU lab time.

## A. Accounts and Access

- [ ] I can log into `hpc4.ust.hk` with my ITSC account.
- [ ] My course repository is cloned on HPC4 (from Lab 2/Lab 3) and I know its local path.
- [ ] I can submit jobs under account `ugiedahpc4`.
- [ ] I understand that shared base models are read-only under `/project/ugiedahpc4/ieda4000i/models/`.

## B. Environment and Package Checks

Run on HPC4 login node:

```bash
source "/opt/shared/.spack-edge/dist/bin/setup-env.sh" -y
conda activate ieda4000i
python --version
```

Expected result:
- Python prints `3.12.x` (or the course-approved version).

Package checks:

```bash
python -c "import torch; print('torch:', torch.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import peft; print('peft:', peft.__version__)"
python -c "import datasets; print('datasets:', datasets.__version__)"
python -c "import accelerate; print('accelerate:', accelerate.__version__)"
python -c "import bitsandbytes as bnb; print('bitsandbytes:', bnb.__version__)"
```

- [ ] All commands run without `ModuleNotFoundError`.

If any package is missing, install in `ieda4000i` and re-check.

## C. Shared Model and Data Availability

```bash
ls /project/ugiedahpc4/ieda4000i/models/
```

- [ ] I can see class models (for example `Qwen3-1.7B`, `Qwen3-4B`).
- [ ] I understand we do **not** need to download base model in advance for standard lab flow.

### Get Lab 4 files onto HPC4

We have added the Lab Files to the course shared folder on HPC4 so you don't need to pull them separately.
Notice that you can access the course repository folder on HPC4 at /project/ugiedahpc4/git/IEDA-4000I-SP26. 
You can copy the Lab 4 files to your home directory on HPC4 using the following commands:

```bash
cp -r /project/ugiedahpc4/ieda4000i/git/IEDA-4000I-SP26/Labs/Lab\ 4 ~/IEDA-4000I-SP26_Lab4
```

- [ ] I have the Lab 4 files in my HPC4 home directory at `~/IEDA-4000I-SP26_Lab4`.
- [ ] `data/` contains `train_lora.jsonl`, `val_lora.jsonl`, and `eval_prompts.jsonl`.
- [ ] `scripts/` contains `train_lora.py`, `train_lora.sbatch`, and `run_eval.py`.

## D. Student-Isolated Path Setup (Required)

Run once before Lab 4 to set up environment variables and folders for your runs. This ensures your outputs and logs are organized and do not conflict with others.:

```bash
export RUN_TAG="${USER}_$(date +%Y%m%d_%H%M%S)"
export LAB4_RUN_ROOT="$HOME/lab4_runs/${RUN_TAG}"
export OUTPUT_DIR="${LAB4_RUN_ROOT}/adapter"
export LOG_DIR="${LAB4_RUN_ROOT}/logs"
export HF_HOME="${LAB4_RUN_ROOT}/hf_cache"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${HF_HOME}"
```

- [ ] I confirmed these folders are created in my own home directory.

## E. SLURM Quick Check (2-minute smoke test)

```bash
sinfo
source "/opt/shared/.spack-edge/dist/bin/setup-env.sh" -y
conda activate ieda4000i
srun --partition=gpu-a30 --gres=gpu:1 --account=ugiedahpc4 --time=00:05:00 --pty bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
exit
```

- [ ] I can get a GPU interactive session.
- [ ] `torch.cuda.is_available()` is `True` on the GPU node.

## F. Bring to Lab 4

- [ ] One short domain-specific prompt set (3-5 prompts).
- [ ] One expected output style (format rules or rubric).
- [ ] One sentence objective for what improvement you want from finetuning.

## In-Class Exit Ticket Preview

You will submit:
1. Base model path used.
2. LoRA training `JOB_ID`.
3. Key lines from `lab4_lora_<jobid>.out`.
4. Adapter output path.
5. Before vs after output comparison on fixed prompts.
6. Short conclusion (what improved and what remains weak).
