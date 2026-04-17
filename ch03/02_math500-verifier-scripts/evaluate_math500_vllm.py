# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch
#
# Evaluate a base model served by vLLM on MATH-500.
# Assumes a vLLM server is already running, e.g.:
#   vllm serve ./hf_models/qwen3-0.6b-base --max-model-len 8192
#
# Uses /v1/completions (base model has no chat template) and runs many
# requests concurrently; vLLM batches them under the hood.

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from reasoning_from_scratch.ch03 import (
    load_math500_test,
    render_prompt,
    extract_final_candidate,
    grade_answer,
    eta_progress_message,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL of the vLLM OpenAI-compatible server.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Served model name. If omitted, auto-detected from /v1/models.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=500,
        help="Number of MATH-500 examples to evaluate (500 = full set).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation (maps to vLLM's max_tokens).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=128,
        help="Number of in-flight requests. B200 can handle large values for 0.6B.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 = greedy (reproducible).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p. Ignored when temperature==0.",
    )
    parser.add_argument(
        "--stop",
        type=str,
        nargs="*",
        default=None,
        help="Optional stop strings passed to vLLM.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=600.0,
        help="Per-request HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output jsonl path. Defaults to math500_<model>-vllm.jsonl.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample correctness while evaluating.",
    )
    return parser.parse_args()


def autodetect_model(base_url):
    """Query /v1/models to get the served-model-name (works when only one is loaded)."""
    url = base_url.rstrip("/") + "/models"
    with httpx.Client(timeout=10.0) as c:
        r = c.get(url, headers={"Authorization": "Bearer unused"})
        r.raise_for_status()
        data = r.json().get("data", [])
    if not data:
        raise RuntimeError(f"No models reported by {url}")
    return data[0]["id"]


async def solve_one(client, model, row, idx, semaphore, sample_kwargs):
    """Send one problem to vLLM, return the record dict."""
    prompt = render_prompt(row["problem"])
    async with semaphore:
        resp = await client.completions.create(
            model=model,
            prompt=prompt,
            **sample_kwargs,
        )
    gen_text = resp.choices[0].text
    extracted = extract_final_candidate(gen_text)
    is_correct = grade_answer(extracted, row["answer"])

    # usage is optional; guard in case a provider doesn't populate it
    completion_tokens = None
    if getattr(resp, "usage", None) is not None:
        completion_tokens = getattr(resp.usage, "completion_tokens", None)

    return {
        "index": idx,
        "problem": row["problem"],
        "gtruth_answer": row["answer"],
        "generated_text": gen_text,
        "extracted": extracted,
        "correct": bool(is_correct),
        "completion_tokens": completion_tokens,
    }


async def run_eval(args, model_name, math_data):
    out_path = Path(
        args.out_path
        or f"math500_{model_name.replace('/', '_')}-vllm.jsonl"
    )

    sample_kwargs = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    if args.temperature > 0.0:
        sample_kwargs["top_p"] = args.top_p
    if args.stop:
        sample_kwargs["stop"] = args.stop

    http_client = httpx.AsyncClient(
        timeout=args.request_timeout,
        limits=httpx.Limits(
            max_connections=max(args.concurrency * 2, 64),
            max_keepalive_connections=max(args.concurrency, 32),
        ),
    )
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key="unused",
        http_client=http_client,
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    num_examples = len(math_data)
    start_time = time.time()

    tasks = [
        asyncio.create_task(
            solve_one(client, model_name, row, i, semaphore, sample_kwargs)
        )
        for i, row in enumerate(math_data, start=1)
    ]

    num_correct = 0
    total_tokens = 0
    records = [None] * num_examples

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for done_count, fut in enumerate(
                asyncio.as_completed(tasks), start=1
            ):
                rec = await fut
                records[rec["index"] - 1] = rec
                num_correct += int(rec["correct"])
                if rec["completion_tokens"]:
                    total_tokens += rec["completion_tokens"]

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                progress_msg = eta_progress_message(
                    processed=done_count,
                    total=num_examples,
                    start_time=start_time,
                    show_eta=True,
                    label="MATH-500 (vLLM)",
                )
                print(progress_msg, end="\r", flush=True)

                if args.verbose:
                    print(
                        f"\n{'='*50}\n{progress_msg}\n{'='*50}\n"
                        f"Index:     {rec['index']}\n"
                        f"Extracted: {rec['extracted']}\n"
                        f"Expected:  {rec['gtruth_answer']}\n"
                        f"Correct so far: {num_correct}/{done_count}\n"
                        f"{'-'*50}"
                    )
    finally:
        await http_client.aclose()

    elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(
        f"\nAccuracy: {acc*100:.2f}% ({num_correct}/{num_examples})"
        f"  |  wall time: {elapsed:.1f}s"
        f"  |  throughput: {num_examples/elapsed:.2f} problems/s"
    )
    if total_tokens:
        print(
            f"Completion tokens: {total_tokens}"
            f"  |  {total_tokens/elapsed:.0f} tok/s"
        )
    print(f"Results written to: {out_path}")
    return num_correct, num_examples, acc


def main():
    args = parse_args()
    model_name = args.model or autodetect_model(args.base_url)

    print(f"Server:      {args.base_url}")
    print(f"Model:       {model_name}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max tokens:  {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")

    math_data = load_math500_test()[: args.dataset_size]
    print(f"Dataset:     {len(math_data)} problems\n")

    asyncio.run(run_eval(args, model_name, math_data))


if __name__ == "__main__":
    main()
