#!/usr/bin/env python3
"""Build a TensorRT-LLM engine for the fine-tuned Mistral 7B.

Three stages, each a separate process invocation so a failure is easy to locate:

  1. **merge**    — fold the QLoRA adapter into the base weights. TensorRT-LLM
                    bakes weights into the engine, so the adapter cannot be a
                    runtime option the way it is under vLLM.
  2. **convert**  — HuggingFace checkpoint -> TensorRT-LLM checkpoint, applying
                    quantization (fp8 on Hopper, int8/int4 AWQ elsewhere).
  3. **build**    — ``trtllm-build`` compiles the checkpoint into a ``.engine``
                    for *this* GPU architecture and these max shapes. The engine
                    is not portable across either.

``--dry-run`` prints the exact commands without executing anything, which is how
the pipeline is verified on a machine without a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a TensorRT-LLM engine.")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--lora-adapter", default=None, help="path to the QLoRA adapter")
    parser.add_argument("--merged-dir", default="artifacts/merged-mistral7b")
    parser.add_argument("--checkpoint-dir", default="artifacts/trtllm-ckpt")
    parser.add_argument("--engine-dir", default="artifacts/trtllm-engine")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--quantization",
        default="fp8",
        choices=["none", "fp8", "int8_wo", "int4_awq"],
        help="fp8 needs Hopper (SM90+); int4_awq is the best fit for Ampere",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-input-len", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-num-tokens", type=int, default=8192)
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "merge", "convert", "build"],
        help="run one stage only",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    return parser


def merge_command(args) -> list[str]:
    """Merge the QLoRA adapter into the base checkpoint via PEFT."""
    script = (
        "import torch;"
        "from peft import PeftModel;"
        "from transformers import AutoModelForCausalLM, AutoTokenizer;"
        f"base=AutoModelForCausalLM.from_pretrained('{args.base_model}',"
        "torch_dtype=torch.bfloat16, device_map='cpu');"
        f"model=PeftModel.from_pretrained(base, '{args.lora_adapter}');"
        "model=model.merge_and_unload();"
        f"model.save_pretrained('{args.merged_dir}');"
        f"AutoTokenizer.from_pretrained('{args.base_model}')"
        f".save_pretrained('{args.merged_dir}')"
    )
    return [sys.executable, "-c", script]


def convert_command(args) -> list[str]:
    source = args.merged_dir if args.lora_adapter else args.base_model
    if args.quantization in ("fp8", "int4_awq"):
        # Calibration-based quantization goes through the ModelOpt quantizer.
        return [
            sys.executable,
            "-m",
            "tensorrt_llm.quantization.quantize_by_modelopt",
            f"--model_dir={source}",
            f"--output_dir={args.checkpoint_dir}",
            f"--dtype={args.dtype}",
            f"--qformat={args.quantization}",
            "--calib_size=512",
            f"--tp_size={args.tp_size}",
        ]
    command = [
        sys.executable,
        "examples/llama/convert_checkpoint.py",
        f"--model_dir={source}",
        f"--output_dir={args.checkpoint_dir}",
        f"--dtype={args.dtype}",
        f"--tp_size={args.tp_size}",
    ]
    if args.quantization == "int8_wo":
        command += ["--use_weight_only", "--weight_only_precision=int8"]
    return command


def build_command(args) -> list[str]:
    return [
        "trtllm-build",
        f"--checkpoint_dir={args.checkpoint_dir}",
        f"--output_dir={args.engine_dir}",
        f"--max_batch_size={args.max_batch_size}",
        f"--max_input_len={args.max_input_len}",
        f"--max_seq_len={args.max_seq_len}",
        f"--max_num_tokens={args.max_num_tokens}",
        # In-flight batching + paged KV are what make the engine competitive with
        # vLLM on throughput rather than only on single-stream latency.
        "--use_paged_context_fmha=enable",
        "--kv_cache_type=paged",
        "--remove_input_padding=enable",
        "--gemm_plugin=auto",
        "--context_fmha=enable",
    ]


def plan(args) -> list[tuple[str, list[str]]]:
    """The commands that would run, in order, for the selected stage."""
    steps: list[tuple[str, list[str]]] = []
    if args.stage in ("all", "merge") and args.lora_adapter:
        steps.append(("merge QLoRA adapter", merge_command(args)))
    if args.stage in ("all", "convert"):
        steps.append(("convert checkpoint", convert_command(args)))
    if args.stage in ("all", "build"):
        steps.append(("build engine", build_command(args)))
    return steps


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quantization == "none":
        args.quantization = None

    steps = plan(args)
    for label, command in steps:
        print(f"\n=== {label} ===")
        print("  " + " ".join(shlex.quote(part) for part in command))
        if args.dry_run:
            continue
        os.makedirs(args.engine_dir, exist_ok=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"stage {label!r} failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode

    if not args.dry_run and args.stage in ("all", "build"):
        manifest = {
            "base_model": args.base_model,
            "lora_adapter": args.lora_adapter,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "tp_size": args.tp_size,
            "max_batch_size": args.max_batch_size,
            "max_input_len": args.max_input_len,
            "max_seq_len": args.max_seq_len,
        }
        path = os.path.join(args.engine_dir, "build_manifest.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
