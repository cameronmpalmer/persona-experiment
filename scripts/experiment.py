#!/usr/bin/env python3
"""Mini replication of "Playing Pretend: Expert Personas Don't Improve
Factual Accuracy" (Prompting Science Report 4, Wharton GAIL Generative AI
Lab; arXiv:2512.05858) on a local model.

Design: 44 graduate-level MMLU-Pro test questions, four system-prompt
conditions, paired design. Every question runs in every condition, so any
accuracy difference is attributable to the system prompt.

Conditions mirror the study's persona ladder:
  baseline  - no persona
  expert    - high-capability persona ("world-class expert... never makes mistakes")
  layperson - neutral low-knowledge persona
  toddler   - negative-capability persona (study: layperson, young child, toddler)

Robustness (added after an Ollama runner wedge 2026-08-13):
  - results are written incrementally after every completed call
  - the run resumes from an existing results file (skips completed tasks)
  - after 3 consecutive request timeouts the run aborts cleanly instead of
    grinding through per-call timeouts

Usage:
  python3 experiment.py qwen3.5:9b
  python3 experiment.py qwen3.5:9b --think

Requires a running Ollama server (http://localhost:11434) with the model
pulled. Sampling is fixed (seed=42), temperature 0. With --think, the
model's chain-of-thought is captured in message.thinking (stored in the
results file) and the final answer is parsed from message.content.
"""

import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA_URL = "http://localhost:11434/api/chat"
QUESTIONS_FILE = "mmlu_pro_sample.json"
THINK_NUM_PREDICT = 2048
REQUEST_TIMEOUT = 120
MAX_CONSECUTIVE_TIMEOUTS = 3

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


def build_user_message(question, options):
    letters = [chr(65 + i) for i in range(len(options))]
    lines = [f"Question: {question}"]
    lines += [f"{letter}) {opt}" for letter, opt in zip(letters, options)]
    return USER_INSTRUCTION + "\n".join(lines)


def run_once(model, question, options, system_prompt, think):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": build_user_message(question, options)})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {"temperature": 0, "num_predict": THINK_NUM_PREDICT if think else 48},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read().decode())
    msg = data["message"]
    return msg["content"].strip(), msg.get("thinking", "")


def parse_letter(text, n_options):
    letters = [chr(65 + i) for i in range(n_options)]
    pat = re.compile(r"\b([" + "|".join(letters) + r"])\b")
    m = pat.search(text)
    return m.group(1) if m else "?"


def make_entry(raw, thinking, n_options, answer_index):
    answer = parse_letter(raw, n_options)
    return {
        "answer": answer,
        "correct": answer == chr(65 + answer_index),
        "raw": raw,
        "thinking": thinking,
    }


def main():
    args = [a for a in sys.argv[1:]]
    think = "--think" in args
    args = [a for a in args if a != "--think"]
    if len(args) < 1:
        print("usage: python3 experiment.py <ollama-model> [--think]", file=sys.stderr)
        sys.exit(1)
    model = args[0]
    slug = model.replace(":", "-").replace("/", "-")
    if think:
        slug += "_think"
    max_workers = 1 if think else 4

    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)
    n = len(questions)
    n_options = len(questions[0]["options"])

    out_path = f"results/results_{slug}.json"

    # Resume: load existing results and skip completed tasks
    results = {cond: {} for cond in CONDITIONS}
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        results = prev.get("results", results)
        print(f"Resuming: {sum(len(v) for v in results.values())} entries already done")

    tasks = []
    for cond, system_prompt in CONDITIONS.items():
        for q in questions:
            if q["id"] in results[cond]:
                continue
            tasks.append((cond, q, system_prompt))

    print(f"Model: {model} [{'THINK' if think else 'NO-THINK'}] | n={n} | "
          f"{len(tasks)} tasks remaining | temp=0 | workers={max_workers}")

    def worker(task):
        cond, q, system_prompt = task
        try:
            raw, thinking = run_once(model, q["question"], q["options"], system_prompt, think)
            return cond, q, make_entry(raw, thinking, len(q["options"]), q["answer_index"]), False
        except Exception as e:
            return cond, q, make_entry(f"ERROR: {e}", "", len(q["options"]), q["answer_index"]), True

    t0 = time.time()
    consecutive_timeouts = 0
    done = 0
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            cond, q, entry, timed_out = fut.result()
            results[cond][q["id"]] = entry
            done += 1
            consecutive_timeouts = consecutive_timeouts + 1 if timed_out else 0

            # Incremental persistence + progress
            if done % 10 == 0 or done == total:
                with open(out_path, "w") as f:
                    json.dump({"model": model, "think": think, "questions": questions,
                               "conditions": CONDITIONS, "results": results}, f, indent=2)
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total - done) / rate if rate > 0 else float("inf")
                print(f"  [{elapsed:5.0f}s] {done}/{total} done ({rate:.2f}/s, ~{remaining/60:.0f}m left)")

            if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                print(f"ABORT: {consecutive_timeouts} consecutive request timeouts - "
                      f"Ollama runner may be wedged. Progress saved; restart the server "
                      f"and rerun to resume.", file=sys.stderr)
                with open(out_path, "w") as f:
                    json.dump({"model": model, "think": think, "questions": questions,
                               "conditions": CONDITIONS, "results": results}, f, indent=2)
                sys.exit(2)

    with open(out_path, "w") as f:
        json.dump({"model": model, "think": think, "questions": questions,
                   "conditions": CONDITIONS, "results": results}, f, indent=2)

    print(f"Model: {model} [{'THINK' if think else 'NO-THINK'}] | MMLU-Pro sample n={n} | "
          f"temp=0 | elapsed {time.time()-t0:.0f}s\n")

    header = f"{'ID':<5}{'Category':<18}{'Ans':<4}"
    for cond in CONDITIONS:
        header += f"{cond:>11}"
    header += f"{'Corr':>5}"
    print(header)
    print("-" * len(header))
    for q in questions:
        correct_letter = chr(65 + q["answer_index"])
        row = f"{q['id']:<5}{q['category']:<18}{correct_letter:<4}"
        n_ok = 0
        for cond in CONDITIONS:
            e = results[cond][q["id"]]
            mark = "OK" if e["correct"] else ("??" if e["answer"] == "?" else "XX")
            if e["correct"]:
                n_ok += 1
            row += f"{mark + ' ' + e['answer']:>11}"
        row += f"{n_ok:>5}/4"
        print(row)

    print()
    print("=== Accuracy by condition ===")
    acc = {}
    for cond in CONDITIONS:
        correct = sum(1 for e in results[cond].values() if e["correct"])
        acc[cond] = correct / n
        print(f"{cond:<10} {correct:>2}/{n} = {acc[cond]:.0%}")

    base = acc["baseline"]
    print()
    print("=== Deltas vs baseline (percentage points) ===")
    for cond in CONDITIONS:
        if cond == "baseline":
            continue
        print(f"{cond:<10} {acc[cond]-base:+.0%}")

    print()
    print("=== Answer changes vs baseline ===")
    for q in questions:
        base_e = results["baseline"][q["id"]]
        changes = []
        for cond in CONDITIONS:
            if cond == "baseline":
                continue
            e = results[cond][q["id"]]
            if e["answer"] != base_e["answer"]:
                changes.append(f"{cond}: {base_e['answer']}->{e['answer']}{' OK' if e['correct'] else ' (wrong)'}")
        if changes:
            print(f"{q['id']} [{q['category']}]: base={base_e['answer']} ({'OK' if base_e['correct'] else 'WRONG'}) | " + " | ".join(changes))


if __name__ == "__main__":
    main()
