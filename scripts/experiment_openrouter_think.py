#!/usr/bin/env python3
"""OpenRouter re-run of the persona experiment WITH thinking enabled.

Why this exists: the local Ollama think run produced a flood of non-answers
because num_predict=2048 capped the ENTIRE completion (thinking + final
answer share the budget). Long-reasoning runs were cut off mid-thought and
returned empty content. On OpenRouter the same model
(qwen/qwen3.5-9b) has a 262k output budget, so we can give thinking and the
answer plenty of room and get a clean read on whether thinking changes the
persona effect.

Design mirrors experiment.py exactly so results are comparable:
  - same 44 MMLU-Pro questions (mmlu_pro_sample.json, seed 42)
  - same four system-prompt conditions (baseline/expert/layperson/toddler)
  - paired: every question answered under every condition
  - temperature 0

Differences from the local run:
  - OpenAI-compatible API at openrouter.ai
  - reasoning enabled (effort=xhigh) + include_reasoning=true so the
    chain-of-thought is captured in message.reasoning
  - max_tokens=16384 (thinking observed at ~2-2.5k tokens locally; this
    leaves massive headroom so the answer always fits)
  - retries with exponential backoff on 429/5xx; abort after 3 consecutive
    failures (mirrors the local wedge protection)

Usage:
  export OPENROUTER_API_KEY=sk-or-...        # also loaded from ~/.bashrc if present
  python3 experiment_openrouter_think.py                    # full 44-question run
  python3 experiment_openrouter_think.py --limit 5          # smoke test, 5 questions
  python3 experiment_openrouter_think.py --workers 4        # tune concurrency

Cost estimate: ~176 calls x ~3k output tokens with thinking ~ $0.08 total
(prompt $0.10/M, completion $0.15/M). Well under a dollar for the full run.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.5-9b"
MAX_TOKENS = 16384          # thinking + answer headroom (model supports 262k)
REQUEST_TIMEOUT = 180
MAX_RETRIES = 5
MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_WORKERS = 20  # measured sweet spot for xhigh reasoning (see README)

QUESTIONS_FILE = "mmlu_pro_sample.json"
OUT_FILE = "results/results_qwen3.5-9b_openrouter_think.json"

CONDITIONS = {
    "baseline": "",
    "expert": "You are a world-class expert in physics, chemistry, biology, computer science, engineering, mathematics, economics, law, and history. You have decades of experience and you never make mistakes.",
    "layperson": "You are a layperson with no specialized training in any technical or academic subject.",
    "toddler": "You are a toddler. You are still learning the basics of the world and have almost no knowledge of science, engineering, law, or mathematics.",
}

USER_INSTRUCTION = (
    "Answer the following multiple-choice question. Reply with only the letter "
    "of the correct answer, e.g. 'A'. Do not include any explanation.\n\n"
)


def load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    # Fall back to ~/.bashrc export line (never printed)
    bashrc = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc):
        with open(bashrc) as f:
            for line in f:
                m = re.match(r"\s*export\s+OPENROUTER_API_KEY\s*=\s*['\"]?([^'\"]+)", line)
                if m:
                    return m.group(1).strip()
    raise SystemExit("OPENROUTER_API_KEY not found in env or ~/.bashrc")


def build_user_message(question, options):
    letters = [chr(65 + i) for i in range(len(options))]
    lines = [f"Question: {question}"]
    lines += [f"{letter}) {opt}" for letter, opt in zip(letters, options)]
    return USER_INSTRUCTION + "\n".join(lines)


def call_openrouter(api_key, model, system_prompt, question, options):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": build_user_message(question, options)})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "seed": 42,
        "reasoning": {"effort": "xhigh", "exclude": False},
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read().decode())
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""
    return content.strip(), reasoning


def run_once(api_key, model, question, options, system_prompt):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_openrouter(api_key, model, system_prompt, question, options)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()
            except Exception:
                pass
            status = e.code
            if status in (429,) or status >= 500:
                last_err = f"HTTP {status}: {body[:200]}"
                time.sleep(min(2 ** attempt, 30))
                continue
            raise
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 30))
            continue
    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_err}")


def parse_letter(text, n_options):
    letters = [chr(65 + i) for i in range(n_options)]
    pat = re.compile(r"\b([" + "|".join(letters) + r"])\b")
    m = pat.search(text)
    return m.group(1) if m else "?"


def make_entry(content, reasoning, n_options, answer_index):
    answer = parse_letter(content, n_options)
    return {
        "answer": answer,
        "correct": answer == chr(65 + answer_index),
        "raw": content,
        "reasoning": reasoning,
    }


def main():
    args = [a for a in sys.argv[1:]]
    limit = None
    workers = DEFAULT_WORKERS
    model = MODEL
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1]); i += 2
        elif a == "--workers" and i + 1 < len(args):
            workers = int(args[i + 1]); i += 2
        elif a == "--model" and i + 1 < len(args):
            model = args[i + 1]; i += 2
        else:
            raise SystemExit(f"unknown arg: {a}")
    if workers < 1:
        raise SystemExit("workers must be >= 1")

    api_key = load_api_key()
    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]
    n = len(questions)
    n_options = len(questions[0]["options"])

    # Resume: load existing results and skip completed tasks
    results = {cond: {} for cond in CONDITIONS}
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            prev = json.load(f)
        results = prev.get("results", results)
        print(f"Resuming: {sum(len(v) for v in results.values())} entries already done", flush=True)

    tasks = []
    for cond, system_prompt in CONDITIONS.items():
        for q in questions:
            if q["id"] in results[cond]:
                continue
            tasks.append((cond, q, system_prompt))

    is_tty = sys.stdout.isatty()
    print(f"Model: {model} [THINK effort=xhigh] | n={n} | {len(tasks)} tasks remaining "
          f"| temp=0 | workers={workers} | max_tokens={MAX_TOKENS}", flush=True)
    if is_tty:
        print("  (Ctrl+C saves progress and exits cleanly; rerun to resume.)", flush=True)

    def worker(task):
        cond, q, system_prompt = task
        try:
            content, reasoning = run_once(api_key, model, q["question"], q["options"], system_prompt)
            return cond, q, make_entry(content, reasoning, len(q["options"]), q["answer_index"]), None
        except Exception as e:
            return cond, q, make_entry(f"ERROR: {e}", "", len(q["options"]), q["answer_index"]), str(e)

    def fmt_eta(seconds):
        if seconds == float("inf"):
            return "???"
        if seconds >= 3600:
            return f"{seconds/3600:.1f}h"
        if seconds >= 60:
            return f"{seconds/60:.0f}m"
        return f"{seconds:.0f}s"

    def render_progress(elapsed, done, total, rate, consecutive_failures, results):
        pct = done / total
        bar_width = 28
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        cond_counts = " ".join(f"{c[:1]}:{len(results[c])}" for c in CONDITIONS)
        if consecutive_failures:
            fails = f" | FAILS={consecutive_failures}"
        else:
            fails = ""
        eta = fmt_eta((total - done) / rate) if rate > 0 else "???"
        return (f"\r  [{elapsed:>5.0f}s] {bar} {done:>3}/{total} ({pct:>5.1%}) "
                f"| {rate:.2f}/s | ~{eta} left | {cond_counts}{fails}")

    t0 = time.time()
    consecutive_failures = 0
    done = 0
    total = len(tasks)
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(worker, t): t for t in tasks}
            for fut in as_completed(futures):
                cond, q, entry, err = fut.result()
                results[cond][q["id"]] = entry
                done += 1
                consecutive_failures = consecutive_failures + 1 if err else 0

                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0

                # Save every 10 completions (crash-safe) and on completion
                if done % 10 == 0 or done == total:
                    save(results, questions, model)

                if is_tty:
                    sys.stdout.write(render_progress(elapsed, done, total, rate,
                                                     consecutive_failures, results))
                    sys.stdout.flush()
                    if done == total:
                        sys.stdout.write("\n")
                elif done % 10 == 0 or done == total:
                    print(f"  [{elapsed:5.0f}s] {done}/{total} done ({rate:.2f}/s, "
                          f"~{fmt_eta((total - done) / rate) if rate > 0 else '???'} left)", flush=True)

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\nABORT: {consecutive_failures} consecutive failures - API may be "
                          f"wedged. Progress saved; rerun to resume.", file=sys.stderr, flush=True)
                    save(results, questions, model)
                    sys.exit(2)
    except KeyboardInterrupt:
        save(results, questions, model)
        print(f"\nInterrupted - progress saved to {OUT_FILE} "
              f"({done}/{total} tasks done). Rerun the same command to resume.", flush=True)
        sys.exit(130)

    save(results, questions, model)
    print(f"\nModel: {model} [THINK] | MMLU-Pro sample n={n} | temp=0 | "
          f"elapsed {time.time()-t0:.0f}s\n")

    print("=== Answers by condition (n=%d) ===" % n)
    acc = {}
    for cond in CONDITIONS:
        entries = results[cond]
        correct = sum(1 for e in entries.values() if e.get("correct"))
        nonans = sum(1 for e in entries.values() if e.get("answer") == "?")
        wrong = len(entries) - correct - nonans
        acc[cond] = correct / n if n else 0
        print(f"  {cond:<10} correct={correct:>2}  wrong={wrong:>2}  non-answer={nonans:>2}  "
              f"({correct/n:.0%} correct)")

    base = acc["baseline"]
    print("\n=== Deltas vs baseline (pp, ? counted as non-answer) ===")
    for cond in CONDITIONS:
        if cond == "baseline":
            continue
        print(f"  {cond:<10} {acc[cond]-base:+.1%}")

    print("\n=== Non-answer reasoning captured (diagnostic) ===")
    for cond in CONDITIONS:
        entries = results[cond]
        na = [e for e in entries.values() if e.get("answer") == "?" and e.get("reasoning", "").strip()]
        print(f"  {cond:<10} {len(na)} non-answers with reasoning text stored")


def save(results, questions, model):
    with open(OUT_FILE, "w") as f:
        json.dump({
            "model": model, "think": True, "api": "openrouter",
            "questions": questions, "conditions": CONDITIONS, "results": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
