"""Quick test: send 4 math problems to the vLLM server with concurrency 2."""

import asyncio
import time
from openai import AsyncOpenAI

BASE_URL = "http://localhost:8000/v1"
MODEL = "./hf_models/qwen3-4b-thinking-2507"

PROMPT_TEMPLATE = (
    "You are a helpful math assistant.\n"
    "Provide a short or medium-length explanation, and then write the "
    "final result on a new line as:\n"
    "\\boxed{{ANSWER}}\n\n"
    "Question:\n{prompt}\n\n"
    "Answer:"
)

PROBLEMS = [
    "What is 17 * 23?",
    "Solve for x: 2x + 5 = 17",
    "What is the derivative of x^3 + 2x^2 - 5x + 3?",
    "If a triangle has sides 3, 4, and 5, what is its area?",
]


async def solve(client, problem, semaphore, idx):
    prompt = PROMPT_TEMPLATE.format(prompt=problem)
    messages = [{"role": "user", "content": prompt}]

    async with semaphore:
        t0 = time.time()
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.0,
        )
        elapsed = time.time() - t0

    choice = response.choices[0].message
    thinking = getattr(choice, "reasoning_content", None) or ""
    content = choice.content or ""

    print(f"\n{'='*60}")
    print(f"Problem {idx+1}: {problem}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Thinking (first 200 chars): {thinking[:200]}")
    print(f"Answer: {content[:300]}")
    return thinking, content, elapsed


async def main():
    client = AsyncOpenAI(base_url=BASE_URL, api_key="unused")
    semaphore = asyncio.Semaphore(2)

    print(f"Sending {len(PROBLEMS)} problems with concurrency=2...")
    t0 = time.time()

    tasks = [solve(client, p, semaphore, i) for i, p in enumerate(PROBLEMS)]
    results = await asyncio.gather(*tasks)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Total wall time: {total:.1f}s for {len(PROBLEMS)} problems")
    print(f"Average per problem: {total/len(PROBLEMS):.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
