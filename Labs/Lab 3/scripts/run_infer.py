#!/usr/bin/env python3
"""Run single-prompt inference with a Hugging Face causal LM model."""

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on HPC4 with a local path or Hugging Face model id."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Local model path or Hugging Face model id, e.g. /project/ugiedahpc4/ieda4000i/models/Qwen3-4B",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="Maximum new tokens"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
        help="Execution device",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable if a model repository requires remote code.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    cuda_available = torch.cuda.is_available()

    if device_arg == "cuda" and not cuda_available:
        print("ERROR: CUDA is not available in this session.")
        print("You are likely on a login node or a CPU-only allocation.")
        print("Request an interactive GPU session first:")
        print(
            "  srun --partition=gpu-l20 --gres=gpu:1 --account=ugiedahpc4 --time=00:30:00 --pty bash"
        )
        sys.exit(2)

    if device_arg == "cpu":
        return "cpu"

    if device_arg == "auto":
        return "cuda" if cuda_available else "cpu"

    return "cuda"


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=== Lab 3 Inference Runner ===")
    print(f"model_path: {args.model_path}")
    print(f"device: {device}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")

    if device == "cuda":
        print(f"gpu_name: {torch.cuda.get_device_name(0)}")

    start = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=args.trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
    except Exception as exc:
        print("ERROR: Failed to load model/tokenizer.")
        print("Possible causes: missing token, missing model permission, wrong model path.")
        print(f"Details: {exc}")
        return 1

    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "temperature": max(args.temperature, 1e-5),
        "top_p": args.top_p,
    }

    if tokenizer.eos_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    try:
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
    except RuntimeError as exc:
        print("ERROR: Generation failed.")
        print(f"Details: {exc}")
        if "out of memory" in str(exc).lower():
            print("Hint: reduce max_new_tokens or use a smaller fallback model.")
        return 1

    generated_ids = output_ids[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    elapsed = time.time() - start
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Generated Text ===")
    print(generated_text if generated_text else "[empty output]")
    print("\n=== Metrics ===")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"generated_tokens: {generated_ids.shape[-1]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

