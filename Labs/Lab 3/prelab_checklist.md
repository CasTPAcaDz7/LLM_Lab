# Lab 3 Lab Activity Checklist (5-Minute Self-Check)

Course: IEDA 4000I  
Lab: Lab 3 - Hugging Face Model Download and Inference on HPC4

Complete this checklist during the lab. The goal is to make sure you have followed the lab activities correctly.

## A. Accounts and Access

- [ ] I can log into `hpc4.ust.hk` with my ITSC account.
- [ ] I have a GitHub account and my course repo is set to **Private**.
- [ ] On my local machine, I have already generated an SSH key.
- [ ] My local SSH public key is already added to GitHub `Settings -> SSH and GPG keys`.
- [ ] I have created a Hugging Face account (Optional).
- [ ] I can access Hugging Face token settings from my account page (Optional).

If you are not sure how to configure SSH key for GitHub, review the related GitHub/SSH parts from previous lab materials first.

## B. HPC4 Environment

Run these commands on HPC4 login node:

```bash
source "/opt/shared/.spack-edge/dist/bin/setup-env.sh" -y
conda activate ieda4000i
python --version
```

Expected result:
- Python prints `3.12.x`.

Package checks:

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
python -c "import accelerate; print(accelerate.__version__)"
python -c "import sentencepiece; print(sentencepiece.__version__)"
```

- [ ] All commands run without `ModuleNotFoundError`.

If any package is missing, run this quick fix in the same `ieda4000i` environment:

```bash
source "/opt/shared/.spack-edge/dist/bin/setup-env.sh" -y
conda activate ieda4000i
pip install -U vllm torch transformers huggingface_hub accelerate sentencepiece
```

If you prefer installing one by one:

```bash
pip install -U vllm
pip install -U torch
pip install -U transformers
pip install -U huggingface_hub
pip install -U accelerate
pip install -U sentencepiece
```

Then re-run the package checks above.

Shared model path check (main class path):

```bash
ls /project/ugiedahpc4/ieda4000i/models/
```

- [ ] I can see `Qwen3-4B` in `/project/ugiedahpc4/ieda4000i/models/`.

## C. Git/GitHub Activity (Required)

Main in-class workflow (SSH-first): create private repo -> local SSH clone -> push -> HPC4 SSH key -> GitHub key registration -> HPC4 SSH clone/pull.

- [ ] Step 1: I created a **Private** repository on GitHub.
- [ ] Step 2: I SSH-cloned the repo to my local machine (`git@github.com:...`).
- [ ] Step 3: I copied `Labs/Lab 3`, then `git add`, `git commit`, and `git push`.
- [ ] Step 4: On HPC4, I generated an SSH key and added it to SSH identity (`ssh-add`).
- [ ] Step 5: I added HPC4 public key to GitHub `Settings -> SSH and GPG keys`.
- [ ] Step 6: On HPC4, I finished `git clone`/`git pull` via SSH and verified `ls "Labs/Lab 3"`.

Reference commands:

```bash
# Local machine (repo root)
git clone git@github.com:YOUR_USERNAME/YOUR_PRIVATE_REPO.git
cd YOUR_PRIVATE_REPO
git remote -v

# copy Canvas Lab 3 folder into Labs/Lab 3 first
git add "Labs/Lab 3"
git commit -m "Add Lab 3 materials"
git push -u origin main

# HPC4
ssh-keygen -t ed25519 -C "YOUR_GITHUB_EMAIL"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub

git clone git@github.com:YOUR_USERNAME/YOUR_PRIVATE_REPO.git
# or: cd ~/YOUR_PRIVATE_REPO && git pull
```

If SSH is blocked on HPC4, use this SSH-over-443 config in `~/.ssh/config`:

```sshconfig
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
```

Then test:

```bash
ssh -T git@github.com
```

For this lab, prefer SSH workflow instead of HTTPS.

## D. Hugging Face Token (Optional for Custom Models)

For Lab 3 main flow (shared model path), this section is optional.
Use this only when you want custom models from Hugging Face Hub.

Create a Read token and log in from HPC4:

```bash
huggingface-cli login
huggingface-cli whoami
```

- [ ] `whoami` shows my account name.

If `huggingface-cli: command not found` appears, install/upgrade it with:

```bash
source "/opt/shared/.spack-edge/dist/bin/setup-env.sh" -y
conda activate ieda4000i
pip install -U huggingface-hub
```

## E. Bring to Lab 3

- [ ] A short prompt I want to test in class.
- [ ] My target base model path note for Lab 4.

## In-Class Exit Ticket Preview

You will submit:
1. Local `git remote -v` proof that remote is SSH (`git@github.com:...`).
2. Local commit hash that adds `Labs/Lab 3`, plus push success proof.
3. HPC4 SSH key setup + GitHub key registration proof.
4. HPC4 `git clone`/`git pull` via SSH + `ls "Labs/Lab 3"` proof.
5. One `run_infer.py` output using `/project/ugiedahpc4/ieda4000i/models/Qwen3-4B`.
6. One `sbatch` job id and corresponding `lab3_infer_<jobid>.out` key lines.
7. (Optional, custom-model track) output of `huggingface-cli whoami`.
