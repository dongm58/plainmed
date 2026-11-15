"""
plainmed/llm.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: The prompted-LLM systems (llm_zeroshot and llm_fewshot).
- PROMPTS: three wordings of the same instruction (A plain request, B a
  health-educator role, C an explicit rule list). All three ask for one
  plain-language paragraph at about an 8th-grade reading level that keeps the
  main findings and how certain they are, and adds no new facts. Running all
  three is the prompt-sensitivity check.
- select_fewshot(): picks the 3 few-shot examples from TRAIN only, with the
  project seed, among pairs short enough to keep the prompt small.
- build_messages() / render_prompt(): the chat messages and the model's own
  chat template. Few-shot examples are given as earlier user/assistant turns.
- clean_output(): strips whitespace and a leading label such as
  "Plain-language summary:" that small models often add. The raw text is kept
  in the output record.
- HFGenerator: batched greedy decoding with Hugging Face transformers. Prompts
  are sorted by length and left-padded so a batch wastes little work.
- StubGenerator: a fake model for tests (returns the first two sentences of
  the abstract), so the whole pipeline runs without a GPU.
- run_system(): writes one JSONL record per example as soon as its batch
  finishes and skips ids already written, so a Colab disconnect only loses
  the batch in progress.

Input: processed examples and the llm settings in config.yaml.
Output: system outputs in the CONTRACT.md format, with timing and token counts in meta.
"""

import random
import re
import time

from plainmed.io import append_jsonl, done_ids
from plainmed.text import split_sentences, word_count

SYSTEM_MESSAGE = "You are a careful assistant that rewrites medical research for patients without changing what the research found."

PROMPTS = {
    "A": (
        "Rewrite the following results from a medical research review as one paragraph of plain language "
        "that a patient with an 8th-grade reading level can understand. Keep the main findings and say how "
        "certain the evidence is. Do not add any facts that are not in the text.\n\nResults:\n{source}"
    ),
    "B": (
        "You are a health educator explaining a research review to a patient. Read the results below and "
        "explain them in a single short paragraph, using everyday words a 13-year-old would know. Tell the "
        "patient what the studies found and how sure the researchers are. Only use information from the "
        "results; do not add anything new.\n\nResults:\n{source}"
    ),
    "C": (
        "Summarize the review results below for the general public. Follow these rules:\n"
        "1. Write exactly one paragraph.\n"
        "2. Aim for an 8th-grade reading level: short sentences and common words.\n"
        "3. Include the main findings.\n"
        "4. Keep the level of certainty (for example, say if the evidence is weak or the results are unclear).\n"
        "5. Leave out statistics unless they are needed to understand a finding.\n"
        "6. Do not add any information that is not in the results.\n\nResults:\n{source}"
    ),
}

LEADING_LABEL = re.compile(
    r"^\s*(?:\*\*)?(?:here(?:'s| is)[^:\n]*|plain[- ]language (?:summary|version)|summary|simplified (?:summary|version))"
    r"(?:\*\*)?\s*(?::|\n)(?:\*\*)?\s*",
    re.IGNORECASE,
)


def select_fewshot(train: list[dict], k: int, seed: int, max_source_words: int = 250, max_target_words: int = 180) -> list[dict]:
    """Pick k training pairs at random (fixed seed) among those short enough for a compact prompt."""
    eligible = [e for e in train if word_count(e["source"]) <= max_source_words and word_count(e["target"]) <= max_target_words]
    rng = random.Random(seed)
    return sorted(rng.sample(eligible, k), key=lambda e: e["id"])


def build_messages(source: str, variant: str, shots: list[dict] | None = None) -> list[dict]:
    """Chat messages for one example: system message, optional few-shot turns, then the real request."""
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for shot in shots or []:
        messages.append({"role": "user", "content": PROMPTS[variant].format(source=shot["source"])})
        messages.append({"role": "assistant", "content": shot["target"]})
    messages.append({"role": "user", "content": PROMPTS[variant].format(source=source)})
    return messages


def render_prompt(messages: list[dict], tokenizer=None) -> str:
    """Turn messages into the model's prompt string with its chat template (a plain template for the stub)."""
    if tokenizer is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages) + "<|assistant|>\n"


def clean_output(text: str) -> str:
    """Strip whitespace and a leading label like 'Plain-language summary:'."""
    text = text.strip()
    cleaned = LEADING_LABEL.sub("", text, count=1).strip()
    return cleaned if cleaned else text


def is_one_paragraph(text: str) -> bool:
    """True if the output is a single paragraph: no blank lines and no bullet or numbered list lines."""
    if re.search(r"\n\s*\n", text.strip()):
        return False
    return not re.search(r"^\s*(?:[-*•]|\d+[.)])\s+", text, re.MULTILINE)


class StubGenerator:
    """Fake model for tests: answers with the first two sentences of the last abstract in the prompt."""

    name = "stub"
    tokenizer = None

    def __init__(self, max_new_tokens: int = 400):
        self.max_new_tokens = max_new_tokens

    def generate(self, prompts: list[str]) -> list[dict]:
        results = []
        for p in prompts:
            source = p.rsplit("Results:\n", 1)[-1].split("\n<|assistant|>")[0]
            text = " ".join(split_sentences(source)[:2])
            results.append({"text": text, "input_tokens": len(p.split()), "output_tokens": len(text.split()), "hit_max_new_tokens": False})
        return results


class HFGenerator:
    """Batched greedy decoding with a Hugging Face causal language model."""

    def __init__(self, model_name: str, max_new_tokens: int = 400, dtype: str = "float16", device: str | None = None,
                 tokenizer=None, model=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype))
        self.model = model.to(self.device).eval()

    def generate(self, prompts: list[str]) -> list[dict]:
        import torch

        batch = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
        new_tokens = out[:, batch["input_ids"].shape[1]:]
        results = []
        stop_ids = {self.tokenizer.pad_token_id, self.tokenizer.eos_token_id}
        stop_ids |= set(self.model.generation_config.eos_token_id if isinstance(self.model.generation_config.eos_token_id, list)
                        else [self.model.generation_config.eos_token_id])
        for i in range(len(prompts)):
            ids = new_tokens[i].tolist()
            length = next((j for j, t in enumerate(ids) if t in stop_ids), len(ids))
            results.append({
                "text": self.tokenizer.decode(ids[:length], skip_special_tokens=True),
                "input_tokens": int(batch["attention_mask"][i].sum()),
                "output_tokens": length,
                "hit_max_new_tokens": length >= self.max_new_tokens,
            })
        return results


def run_system(examples: list[dict], system: str, variant: str, shots: list[dict], generator, out_path,
               batch_size: int = 8, log_every: int = 1) -> None:
    """Generate outputs for every example not already in out_path, appending records batch by batch."""
    finished = done_ids(out_path)
    pending = [e for e in examples if e["id"] not in finished]
    prompts = {e["id"]: render_prompt(build_messages(e["source"], variant, shots), generator.tokenizer) for e in pending}
    pending.sort(key=lambda e: len(prompts[e["id"]]))
    shot_ids = [s["id"] for s in shots]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        t0 = time.time()
        results = generator.generate([prompts[e["id"]] for e in batch])
        seconds = (time.time() - t0) / len(batch)
        for e, r in zip(batch, results):
            output = clean_output(r["text"])
            append_jsonl(out_path, {
                "id": e["id"], "system": system, "output": output,
                "meta": {"model": generator.name, "variant": variant, "fewshot_ids": shot_ids,
                         "seconds": round(seconds, 3), "input_tokens": r["input_tokens"],
                         "output_tokens": r["output_tokens"], "hit_max_new_tokens": r["hit_max_new_tokens"],
                         "one_paragraph": is_one_paragraph(output), "raw_output": r["text"]},
            })
        batch_number = start // batch_size + 1
        if log_every and batch_number % log_every == 0:
            done = len(finished) + start + len(batch)
            print(f"{system} {variant}: {done}/{len(examples)} done, {seconds:.2f} s per example in this batch", flush=True)
