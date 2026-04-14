"""
Re-evaluate answers in a JSON file against ground truth using the same
grading logic as evaluate_json.py and evaluate_math500.py (i.e.,
extract_final_candidate + grade_answer from reasoning_from_scratch.ch03).

Prints a per-entry report showing which answers are correct and which are
wrong, then prints overall accuracy. Optionally updates the JSON with
corrected _extracted_answer and _is_correct fields.

Example usage::

    # Check existing answers (message_content field) against gtruth_answer
    python recheck_answers.py --json_path problems.json

    # Check newly generated answers from vLLM
    python recheck_answers.py --json_path problems_with_new_answers.json \
        --gen_key message_content_new

    # Also update the JSON with corrected _extracted_answer / _is_correct
    python recheck_answers.py --json_path problems.json --update

    # Only show wrong answers (less noisy)
    python recheck_answers.py --json_path problems.json --wrong_only
"""

import argparse
import json
from pathlib import Path

from reasoning_from_scratch.ch03 import (
    extract_final_candidate,
    grade_answer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Re-check answers in a JSON file against ground truth.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the JSON or JSONL file.",
    )
    parser.add_argument(
        "--gt_key",
        type=str,
        default="gtruth_answer",
        help="Key for the ground-truth answer.",
    )
    parser.add_argument(
        "--gen_key",
        type=str,
        default="message_content",
        help="Key for the generated text to extract the answer from.",
    )
    parser.add_argument(
        "--wrong_only",
        action="store_true",
        help="Only print entries that are incorrect.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Update the JSON file in place with corrected "
            "_extracted_answer and _is_correct fields."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Write updated JSON to this path instead of overwriting the input. "
            "Only used with --update."
        ),
    )
    return parser.parse_args()


def load_records(json_path):
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            parsed = json.load(f)
        except json.JSONDecodeError:
            # Try JSONL
            f.seek(0)
            records = []
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} in {path}: {exc}"
                    ) from exc
            return records

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if "records" in parsed and isinstance(parsed["records"], list):
            return parsed["records"]
        return [parsed]

    raise ValueError(
        f"Unsupported JSON root type in {path}: {type(parsed).__name__}"
    )


def main():
    args = parse_args()

    records = load_records(args.json_path)
    total = len(records)
    num_correct = 0
    wrong_indices = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            print(f"[{idx}] Skipping non-dict record.")
            continue

        gt = record.get(args.gt_key)
        gen_text = record.get(args.gen_key)

        if gt is None:
            print(f"[{idx}] Missing ground truth key '{args.gt_key}', skipping.")
            continue
        if gen_text is None:
            print(f"[{idx}] Missing generated text key '{args.gen_key}', skipping.")
            continue

        extracted = extract_final_candidate(gen_text)
        is_correct = grade_answer(extracted, gt)

        if args.update:
            record["_extracted_answer"] = extracted
            record["_is_correct"] = is_correct

        if is_correct:
            num_correct += 1
            if not args.wrong_only:
                print(f"[{idx}] CORRECT  | extracted: {extracted} | gt: {gt}")
        else:
            wrong_indices.append(idx)
            problem_preview = record.get("problem", "")[:80]
            print(
                f"[{idx}] WRONG    | extracted: {extracted} | gt: {gt} "
                f"| problem: {problem_preview}..."
            )

    acc = num_correct / total if total else 0.0
    print(f"\n{'='*60}")
    print(f"Accuracy: {acc*100:.1f}% ({num_correct}/{total})")
    print(f"Wrong: {len(wrong_indices)} entries at indices: {wrong_indices}")

    if args.update:
        out_path = (
            Path(args.output_json) if args.output_json
            else Path(args.json_path)
        )
        tmp = out_path.with_name(f"{out_path.name}.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp.replace(out_path)
        print(f"Updated JSON written to: {out_path}")


if __name__ == "__main__":
    main()
