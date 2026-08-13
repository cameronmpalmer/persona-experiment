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

Usage:
  python3 experiment.py qwen3.5:9b
  python3 experiment.py qwen3.5:0.8b

Requires a running Ollama server (http://localhost:11434) with the model
pulled. Sampling is fixed (seed=42), temperature 0, thinking disabled.
"""

import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OLLAMA_URL = "http://localhost:11434/api/chat"
MAX_WORKERS = 4
QUESTIONS_FILE = "mmlu_pro_sample.json"

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


def run_once(model, question, options, system_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": build_user_message(question, options)})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 48},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    return data["message"]["content"].strip()


def parse_letter(text, n_options):
    letters = [chr(65 + i) for i in range(n_options)]
    pat = re.compile(r"\b([" + "|".join(letters) + r"])\b")
    m = pat.search(text)
    return m.group(1) if m else "?"


def main():
    if len(sys.argv) < 2:
        print("usage: python3 experiment.py <ollama-model>", file=sys.stderr)
        sys.exit(1)
    model = sys.argv[1]
    slug = model.replace(":", "-").replace("/", "-")

    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)
    n = len(questions)
    n_options = len(questions[0]["options"])

    tasks = []
    for cond, system_prompt in CONDITIONS.items():
        for q in questions:
            tasks.append((cond, q, system_prompt))

    def worker(task):
        cond, q, system_prompt = task
        try:
            raw = run_once(model, q["question"], q["options"], system_prompt)
        except Exception as e:
            raw = f"ERROR: {e}"
        answer = parse_letter(raw, len(q["options"]))
        correct_letter = chr(65 + q["answer_index"])
        return cond, q, {
            "answer": answer,
            "correct": answer == correct_letter,
            "raw": raw,
        }

    results = {cond: {} for cond in CONDITIONS}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for cond, q, entry in ex.map(worker, tasks):
            results[cond][q["id"]] = entry

    out_path = f"results/results_{slug}.json"
    with open(out_path, "w") as f:
        json.dump({"model": model, "questions": questions,
                   "conditions": CONDITIONS, "results": results},
                  f, indent=2)

    print(f"Model: {model} | MMLU-Pro sample n={n} | temp=0 | elapsed {time.time()-t0:.0f}s\n")

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
