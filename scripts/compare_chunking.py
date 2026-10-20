"""Compare chunking strategies x scrapers x retrieval modes on the labelled questions.

Writes docs/chunking_comparison.md and data/experiments/chunking_scores.json.
Usage:  python scripts/compare_chunking.py [--configs requests:structured,requests:recursive,...]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from statistics import mean

import yaml

from placidway_bot.config import DATA_DIR, RAW_DIR, ROOT_DIR, get_settings, load_sources
from placidway_bot.ingest.chunker import STRATEGIES, chunk_plain_text
from placidway_bot.ingest.models import PageDoc
from placidway_bot.retrieval.hybrid import retrieve
from placidway_bot.retrieval.store import build_index
from placidway_bot.retrieval.voyage_service import VoyageService

LABELS = ROOT_DIR / "eval" / "retrieval_labels.yaml"
INDEX_ROOT = DATA_DIR / "experiments" / "index"
OUT_MD = ROOT_DIR / "docs" / "chunking_comparison.md"
OUT_JSON = DATA_DIR / "experiments" / "chunking_scores.json"
DEFAULT_CONFIGS = ["requests:recursive", "requests:structured"]
MODES = ["bm25", "vector", "hybrid", "hybrid_rerank"]
COHERE_MODELS = {"hybrid_rerank_cohere_pro": "rerank-v4.0-pro", "hybrid_rerank_cohere_fast": "rerank-v4.0-fast"}
RERANK_MODES = ("hybrid_rerank", *COHERE_MODELS)
RERANKER_NAME = {"hybrid_rerank": "Voyage rerank-3", "hybrid_rerank_cohere_pro": "Cohere rerank-v4.0-pro",
                 "hybrid_rerank_cohere_fast": "Cohere rerank-v4.0-fast"}
K = 5
log = logging.getLogger("compare_chunking")


LANG_NAME = {"en": "English", "es": "Spanish", "ar": "Arabic"}


def load_questions() -> list[dict]:
    """Base questions plus their Spanish/Arabic translations (same facts, id suffixed with @es / @ar)."""
    out = []
    for q in yaml.safe_load(LABELS.read_text(encoding="utf-8"))["questions"]:
        out.append({**q, "lang": "en"})
        for lang, text in (q.get("translations") or {}).items():
            out.append({**q, "id": f"{q['id']}@{lang}", "question": text, "lang": lang})
    return out


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"(?<=\d),(?=\d)", "", text.lower()))


def satisfies(chunk, fact: dict) -> bool:
    return chunk.slug in fact["slugs"] and all(norm(s) in norm(chunk.text) for s in fact["contains"])


def load_chunks(method: str, strategy: str):
    chunks = []
    for src in load_sources():
        doc = PageDoc.from_dict(json.loads((DATA_DIR / "experiments" / "requests" / f"{src.slug}.json")
                                           .read_text(encoding="utf-8")))
        if method == "chrome":
            text = (RAW_DIR / "chrome" / f"{src.slug}.txt").read_text(encoding="utf-8")
            chunks += chunk_plain_text(src.slug, src.url, doc.title, src.page_type, text, strategy=strategy)
        else:
            doc = PageDoc.from_dict(json.loads((DATA_DIR / "experiments" / method / f"{src.slug}.json")
                                               .read_text(encoding="utf-8")))
            chunks += STRATEGIES[strategy](doc)
    return chunks


def score_question(hits, q: dict) -> dict:
    facts = q["facts"]
    top = hits[:K]
    first = next((r + 1 for r, h in enumerate(hits) if any(satisfies(h.chunk, f) for f in facts)), None)
    found = [any(satisfies(h.chunk, f) for h in top) for f in facts]
    return dict(
        fact_recall=mean(found) if facts else None,
        all_facts=all(found) if facts else None,
        mrr=(1 / first) if first else 0.0,
        first_relevant_rank=first,
        relevant_score=hits[first - 1].score if first else None,
        top1_score=hits[0].score if hits else 0.0,
        top=[f"{h.chunk.chunk_id} ({h.score:.3f})" for h in top],
    )


def evaluate(configs: list[str], rerank_configs: set[str], only_modes: set[str] = frozenset()) -> dict:
    voyage = VoyageService()
    cohere = {}
    if get_settings().cohere_api_key:
        from placidway_bot.retrieval.cohere_reranker import CohereReranker
        cohere = {mode: CohereReranker(model=model) for mode, model in COHERE_MODELS.items()}
    questions = load_questions()
    voyage.embed([q["question"] for q in questions], "query")          # one batched call, then cached
    results: dict = {}
    for cfg in configs:
        method, strategy = cfg.split(":")
        chunks = load_chunks(method, strategy)
        t0 = time.perf_counter()
        index = build_index(f"{method}__{strategy}", chunks, voyage, root=INDEX_ROOT)
        log.info("%s: %d chunks indexed in %.0fs", cfg, len(chunks), time.perf_counter() - t0)
        results[cfg] = {"n_chunks": len(chunks), "avg_chars": round(mean(len(c.text) for c in chunks)), "modes": {}}
        modes = MODES if cfg in rerank_configs else MODES[:-1]
        if cohere and cfg in rerank_configs:
            modes = [*modes, *cohere]
        if only_modes:
            modes = [m for m in modes if m in only_modes]
        for mode in modes:
            per_q = {}
            for q in questions:
                if mode in cohere:
                    hits = retrieve(index, voyage, q["question"], mode="hybrid_rerank", k=10, reranker=cohere[mode])
                else:
                    hits = retrieve(index, voyage, q["question"], mode=mode, k=10)
                per_q[q["id"]] = score_question(hits, q) | {"type": q["type"], "lang": q["lang"]}
            results[cfg]["modes"][mode] = per_q
            log.info("  %-14s done", mode)
    return results


def summarize(per_q: dict) -> dict:
    pos = [v for v in per_q.values() if v["fact_recall"] is not None]
    neg = [v for v in per_q.values() if v["fact_recall"] is None]
    return dict(fact_recall=100 * mean(v["fact_recall"] for v in pos),
                all_facts=100 * mean(v["all_facts"] for v in pos),
                mrr=mean(v["mrr"] for v in pos),
                pos_top1=mean(v["top1_score"] for v in pos),
                neg_top1_max=max((v["top1_score"] for v in neg), default=0.0),
                pos_top1_min=min(v["top1_score"] for v in pos))


def calibrate(per_q: dict) -> float:
    """Top-1 score threshold that best separates answerable from unanswerable questions (all languages)."""
    pos = sorted(v["top1_score"] for v in per_q.values() if v["fact_recall"] is not None)
    neg = sorted(v["top1_score"] for v in per_q.values() if v["fact_recall"] is None)
    candidates = sorted(set(pos + neg))
    best = max(candidates, key=lambda t: sum(p >= t for p in pos) + sum(n < t for n in neg))
    below = [c for c in candidates if c < best]
    return (best + below[-1]) / 2 if below else best


def pdf_pass(v: dict, threshold: float) -> bool:
    if v["fact_recall"] is None:              # must not be answered from the content
        return v["top1_score"] < threshold
    return bool(v["all_facts"]) and v["top1_score"] >= threshold


def _modes(results: dict) -> list[tuple[str, str]]:
    modes = [("vector", "no reranker (vector)"), *((m, RERANKER_NAME[m]) for m in RERANK_MODES)]
    return [(m, label) for m, label in modes if any(m in r["modes"] for r in results.values())]


def pdf_section(results: dict, questions: list[dict]) -> list[str]:
    modes = _modes(results)
    thresholds = {(cfg, m): calibrate(results[cfg]["modes"][m]) for cfg in results for m, _ in modes
                  if m in results[cfg]["modes"]}
    pdf = [q for q in questions if q["id"].startswith("pdf")]
    langs = [lang for lang in LANG_NAME if any(q["lang"] == lang for q in pdf)]
    lines = ["", "## The PDF test questions (English, Spanish, Arabic)", "",
             "Pass = every required fact is in the top-5 *and* the top score clears the \"I don't know\" threshold; "
             "for questions that must not be answered, pass = top score stays below it. One threshold per strategy "
             "and mode, calibrated on all questions in all languages. Score = relevance of the first correct chunk "
             "(cosine similarity without a reranker, the reranker's relevance score with one); for must-not-answer "
             "questions it is the top-1 score.", "",
             "### Passed per language", "",
             "| Strategy | Mode | Threshold | " + " | ".join(LANG_NAME[x] for x in langs) + " | All |",
             "|---|---|---|" + "---|" * (len(langs) + 1)]
    for cfg in results:
        for m, label in modes:
            if m not in results[cfg]["modes"]:
                continue
            per_q, thr = results[cfg]["modes"][m], thresholds[(cfg, m)]
            counts = []
            for lang in langs:
                qs = [q for q in pdf if q["lang"] == lang]
                counts.append(f"{sum(pdf_pass(per_q[q['id']], thr) for q in qs)}/{len(qs)}")
            total = sum(pdf_pass(per_q[q["id"]], thr) for q in pdf)
            lines.append(f"| {cfg.split(':')[1]} | {label} | {thr:.3f} | " + " | ".join(counts)
                         + f" | **{total}/{len(pdf)}** |")
    for lang in langs:
        lines += ["", f"### {LANG_NAME[lang]}", "",
                  "| Question | " + " | ".join(f"{cfg.split(':')[1]} - {label}" for cfg in results for _, label in modes)
                  + " |", "|---|" + "---|" * (len(results) * len(modes))]
        for q in (q for q in pdf if q["lang"] == lang):
            cells = []
            for cfg in results:
                for m, _ in modes:
                    if m not in results[cfg]["modes"]:
                        cells.append("n/a")
                        continue
                    v = results[cfg]["modes"][m][q["id"]]
                    score = v["top1_score"] if v["fact_recall"] is None else v["relevant_score"]
                    mark = "PASS" if pdf_pass(v, thresholds[(cfg, m)]) else "FAIL"
                    cells.append(f"{mark} ({score:.3f})" if score is not None else f"{mark} (not retrieved)")
            lines.append(f"| {q['id'].split('@')[0]}: {q['question'][:50]} | " + " | ".join(cells) + " |")
    return lines


def render(results: dict, questions: list[dict]) -> str:
    out = ["# Chunking & retrieval comparison", "",
           f"Labelled set: `eval/retrieval_labels.yaml` ({len(questions)} questions incl. Spanish/Arabic versions of "
           f"the PDF tests). {sum(1 for q in questions if q['facts'])} answerable "
           f"questions (the PDF's 8 test types filled in for our pages plus hidden-style, multilingual and trick "
           f"questions) and {sum(1 for q in questions if not q['facts'])} that must stay unanswered. "
           "A fact counts as retrieved when a top-5 chunk comes from the right page **and** contains the required text.",
           "Generated by `scripts/compare_chunking.py`.", "",
           "## Summary (top-5)", "",
           "| Scraper : chunking | Chunks | Avg chars | Mode | Fact recall@5 | All facts@5 | MRR@10 |",
           "|---|---|---|---|---|---|---|"]
    for cfg, r in results.items():
        for mode, per_q in r["modes"].items():
            s = summarize(per_q)
            out.append(f"| {cfg} | {r['n_chunks']} | {r['avg_chars']} | {mode} | {s['fact_recall']:.1f}% | "
                       f"{s['all_facts']:.1f}% | {s['mrr']:.3f} |")
    out += ["", "## Reranker relevance scores (hybrid + rerank)", "",
            "Separation between answerable and unanswerable questions decides the \"I don't know\" threshold.", "",
            "| Scraper : chunking | Mean top-1 (answerable) | Min top-1 (answerable) | Max top-1 (unanswerable) |",
            "|---|---|---|---|"]
    reranked = [cfg for cfg, r in results.items() if any(m in r["modes"] for m in RERANK_MODES)]
    for cfg in reranked:
        for mode in RERANK_MODES:
            if mode in results[cfg]["modes"]:
                s = summarize(results[cfg]["modes"][mode])
                out.append(f"| {cfg} ({RERANKER_NAME[mode]}) | {s['pos_top1']:.3f} | {s['pos_top1_min']:.3f} | "
                           f"{s['neg_top1_max']:.3f} |")
    out += pdf_section(results, questions)
    cols = [(cfg, m) for cfg in reranked for m in RERANK_MODES if m in results[cfg]["modes"]]
    out += ["", "## Per question (all languages): facts found in top-5 (top-1 score)", "",
            "| Question | Type | " + " | ".join(f"{c.split(':')[1]} - {RERANKER_NAME[m]}" for c, m in cols) + " |",
            "|---|---|" + "---|" * len(cols)]
    for q in questions:
        cells = []
        for cfg, m in cols:
            v = results[cfg]["modes"][m][q["id"]]
            cells.append(f"top1 {v['top1_score']:.2f}" if v["fact_recall"] is None
                         else f"{v['fact_recall'] * 100:.0f}% ({v['top1_score']:.2f})")
        out.append(f"| {q['id']}: {q['question'][:50]} | {q['type']} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--rerank-configs", default="requests:recursive,requests:structured",
                        help="configs that also get the (rate-limited) rerank stage")
    parser.add_argument("--modes", default="", help="comma list to run only some modes, e.g. hybrid_rerank_cohere_pro")
    parser.add_argument("--tag", default="", help="write results to *_<tag> files instead of the main report")
    args = parser.parse_args()
    results = evaluate(args.configs.split(","), set(args.rerank_configs.split(",")),
                       set(filter(None, args.modes.split(","))))
    if args.tag:
        global OUT_JSON, OUT_MD
        OUT_JSON = OUT_JSON.with_name(f"chunking_scores_{args.tag}.json")
        OUT_MD = OUT_MD.with_name(f"chunking_comparison_{args.tag}.md")
    OUT_JSON.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    questions = load_questions()
    marker = "<!-- AUTO-GENERATED ABOVE -->"
    tail = OUT_MD.read_text(encoding="utf-8").split(marker, 1)[1] if OUT_MD.exists() and marker in OUT_MD.read_text(encoding="utf-8") else ""
    OUT_MD.write_text(render(results, questions) + "\n" + marker + tail, encoding="utf-8")
    print(render(results, questions))


if __name__ == "__main__":
    main()
