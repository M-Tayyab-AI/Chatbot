"""Evaluation runner: LangSmith datasets + experiments, a local JSON copy of every run, and the results report.

    python scripts/run_eval.py retrieval                          # chunking A vs B (Voyage + Cohere only, no Gemini)
    python scripts/run_eval.py e2e --suite pdf --temperature 0.2  # the 8 brief tests through the whole bot
    python scripts/run_eval.py e2e --suite extended               # other languages, follow-ups, unknown topic
    python scripts/run_eval.py report                             # -> eval/results/test_results.md

Every run is traced to LangSmith (each graph node, router decision, prompt, token count and latency) and logged as an
experiment on a dataset, so runs can be compared side by side there. Checks are deterministic code
(placidway_bot.evaluation.checks), not an LLM judge. The analyze step is cached (data/eval_cache/) so the free-tier
Gemini quota is spent on the answer model the temperature sweep measures.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from placidway_bot.config import DATA_DIR, get_settings, load_sources  # noqa: E402
from placidway_bot.evaluation.checks import Check, e2e_checks, passed_all, retrieval_checks  # noqa: E402
from placidway_bot.guardrails.price_check import all_numbers  # noqa: E402
from placidway_bot.llm import configure_tracing  # noqa: E402

CASES_FILE = ROOT / "eval" / "test_questions.yaml"
LABELS_FILE = ROOT / "eval" / "retrieval_labels.yaml"
RUNS_DIR = ROOT / "eval" / "results" / "runs"
REPORT = ROOT / "eval" / "results" / "test_results.md"
LLM_CACHE = DATA_DIR / "eval_cache" / "llm_cache.sqlite"
EXPERIMENT_INDEX_ROOT = DATA_DIR / "experiments" / "index"
DATASETS = {"pdf": "placidway-pdf-tests", "extended": "placidway-extended-tests", "retrieval": "placidway-retrieval"}
# Gemini 503 "high demand" / 429 per-minute errors are the provider's, not the bot's: rerun the case
MODEL_ERROR_RETRIES = 3
STRATEGY_LABEL = {"recursive": "A. recursive 1000/200", "structured": "B. structured (production)"}
log = logging.getLogger("run_eval")


# ---------------------------------------------------------------------------------------------- datasets
def load_cases(suite: str) -> list[dict]:
    return yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))[suite]


def load_retrieval_questions() -> list[dict]:
    """Labelled questions plus their Spanish/Arabic translations (same facts), as in compare_chunking.py."""
    out = []
    for q in yaml.safe_load(LABELS_FILE.read_text(encoding="utf-8"))["questions"]:
        out.append({**q, "lang": "en"})
        for lang, text in (q.get("translations") or {}).items():
            out.append({**q, "id": f"{q['id']}@{lang}", "question": text, "lang": lang})
    return out


def sync_dataset(client, name: str, description: str, examples: list[dict]) -> None:
    """Create or update a LangSmith dataset so it matches `examples` ({id, inputs, outputs, metadata})."""
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(name, description=description)
    existing = {e.metadata.get("case_id"): e for e in client.list_examples(dataset_id=dataset.id)}
    for ex in examples:
        metadata = {**ex.get("metadata", {}), "case_id": ex["id"]}
        old = existing.pop(ex["id"], None)
        if old is None:
            client.create_example(inputs=ex["inputs"], outputs=ex["outputs"], metadata=metadata, dataset_id=dataset.id)
        elif old.inputs != ex["inputs"] or old.outputs != ex["outputs"]:
            client.update_example(old.id, inputs=ex["inputs"], outputs=ex["outputs"], metadata=metadata)
    if existing:
        client.delete_examples([e.id for e in existing.values()])
    log.info("dataset %s: %d examples", name, len(examples))


# ---------------------------------------------------------------------------------------------- shared helpers
class _NoGapLog:
    """Evaluation questions must not land in the production unanswered-question log."""

    def log(self, **_) -> None:
        pass


def _as_langsmith(checks: list[Check]) -> dict:
    results = [{"key": c.key, "score": float(c.score), "comment": c.comment} for c in checks if c.score is not None]
    # pass = every pass/fail check passed (MRR is a ranking quality score, not a pass/fail check)
    results.append({"key": "pass", "score": float(all(r["score"] == 1.0 for r in results
                                                      if not r["key"].startswith("mrr")))})
    return {"results": results}


def _save_run(payload: dict) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{payload['experiment']}.json"
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("saved %s", path.relative_to(ROOT))
    return path


def _rows(results) -> list[dict]:
    rows = []
    for r in results:
        run, example = r["run"], r["example"]
        evals = {e.key: {"score": e.score, "comment": e.comment} for e in r["evaluation_results"]["results"]}
        latency = (run.end_time - run.start_time).total_seconds() if run.end_time and run.start_time else None
        rows.append({"case_id": example.metadata.get("case_id"), "inputs": example.inputs, "outputs": run.outputs,
                     "error": run.error, "latency_s": latency, "checks": evals})
    return sorted(rows, key=lambda x: x["case_id"] or "")


# ---------------------------------------------------------------------------------------------- end-to-end
def build_bot(temperature: float, model: str):
    """The production graph with: chosen answer model + temperature, no fallback model (a quota failure must show
    as an error, not be hidden by Flash-Lite), no hidden client retries (failed calls still count toward the daily
    quota; target() reruns a failed case after a pause), cached analyze step, in-memory history, no production logs."""
    from langgraph.checkpoint.memory import InMemorySaver

    from placidway_bot.chat import ChatBot
    from placidway_bot.evaluation.llm_cache import SQLiteLLMCache
    from placidway_bot.graph.nodes import Services
    from placidway_bot.graph.state import Analysis, GroundedAnswer
    from placidway_bot.llm import chat_model
    from placidway_bot.retrieval.retriever import KnowledgeRetriever

    base = get_settings()
    settings = base.model_copy(update={"gemini_temperature": temperature, "gemini_model": model})
    analyzer_llm = chat_model("low", settings=base, model=base.gemini_analyze_model)
    analyzer_llm.cache = SQLiteLLMCache(LLM_CACHE)
    services = Services(analyzer=analyzer_llm.with_structured_output(Analysis),
                        answerer=chat_model(settings=settings, model=model, max_retries=0)
                        .with_structured_output(GroundedAnswer),
                        retriever=KnowledgeRetriever(settings=settings), gap_logger=_NoGapLog(), settings=settings)
    return ChatBot(services=services, checkpointer=InMemorySaver())


def run_e2e(client, suite: str, temperature: float, model: str, pause: float) -> None:
    cases = load_cases(suite)
    sync_dataset(client, DATASETS[suite], f"PlacidWay chatbot end-to-end cases ({suite}), eval/test_questions.yaml",
                 [{"id": c["id"], "inputs": {"question": c["question"], "history": c.get("history") or []},
                   "outputs": {"expect": c["expect"]}, "metadata": {"test": c["test"]}} for c in cases])
    chunks = json.loads(_active_chunks_path().read_text(encoding="utf-8"))
    kb_numbers = set().union(*(all_numbers(c["text"]) for c in chunks))
    urls_by_slug = {s.slug: s.url for s in load_sources()}
    bot = build_bot(temperature, model)

    def target(inputs: dict) -> dict:
        history = [tuple(turn) for turn in inputs.get("history") or []]
        for attempt in range(MODEL_ERROR_RETRIES + 1):
            time.sleep(pause if attempt == 0 else 30 * attempt)   # free tier: 5 requests/minute per model
            out = asdict(bot.answer(inputs["question"], f"eval-{uuid.uuid4().hex[:8]}", history=history))
            if out["status"] != "error":
                break
            log.warning("model error (503 overload / 429 rate limit); rerunning the case")
        return out | {"model_error_retries": attempt}

    def evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        case = {"question": inputs["question"], "expect": reference_outputs["expect"]}
        return _as_langsmith(e2e_checks(outputs, case, kb_numbers, urls_by_slug))

    tag = f"T{temperature:g}"
    results = client.evaluate(target, data=DATASETS[suite], evaluators=[evaluator],
                              experiment_prefix=f"e2e-{suite}-{model}-{tag}", max_concurrency=1,
                              metadata={"suite": suite, "model": model, "temperature": temperature,
                                        "analyze_model": get_settings().gemini_analyze_model,
                                        "reranker": get_settings().cohere_rerank_model,
                                        "threshold": get_settings().relevance_threshold})
    rows = _rows(results)
    _save_run({"kind": "e2e", "suite": suite, "model": model, "temperature": temperature, "pause": pause,
               "experiment": results.experiment_name, "created_at": datetime.now(timezone.utc).isoformat(),
               "tests": {c["id"]: c["test"] for c in cases}, "rows": rows})
    passed = sum(r["checks"].get("pass", {}).get("score") == 1.0 for r in rows)
    print(f"{results.experiment_name}: {passed}/{len(rows)} cases passed every check")


def _active_chunks_path() -> Path:
    manifest = json.loads((DATA_DIR / "index" / "manifest.json").read_text(encoding="utf-8"))
    return DATA_DIR / "index" / manifest["active_index"] / "chunks.json"


# ---------------------------------------------------------------------------------------------- retrieval
def run_retrieval(client, strategies: list[str]) -> None:
    from placidway_bot.retrieval.cohere_reranker import CohereReranker
    from placidway_bot.retrieval.hybrid import retrieve
    from placidway_bot.retrieval.store import load_index
    from placidway_bot.retrieval.voyage_service import VoyageService

    settings = get_settings()
    questions = load_retrieval_questions()
    sync_dataset(client, DATASETS["retrieval"], "Labelled retrieval questions (EN + ES/AR), eval/retrieval_labels.yaml",
                 [{"id": q["id"], "inputs": {"question": q["question"]}, "outputs": {"facts": q["facts"]},
                   "metadata": {"type": q["type"], "lang": q["lang"]}} for q in questions])
    voyage, reranker = VoyageService(), CohereReranker()
    for strategy in strategies:
        index = load_index(f"requests__{strategy}", root=EXPERIMENT_INDEX_ROOT)

        def target(inputs: dict) -> dict:
            hits = retrieve(index, voyage, inputs["question"], mode="hybrid_rerank", k=10, reranker=reranker)
            return {"hits": [{"chunk_id": h.chunk.chunk_id, "slug": h.chunk.slug, "section": h.chunk.section,
                              "score": round(h.score, 4), "text": h.chunk.text} for h in hits]}

        def evaluator(outputs: dict, reference_outputs: dict) -> dict:
            return _as_langsmith(retrieval_checks(outputs["hits"], reference_outputs["facts"],
                                                  settings.relevance_threshold))

        results = client.evaluate(target, data=DATASETS["retrieval"], evaluators=[evaluator],
                                  experiment_prefix=f"retrieval-{strategy}", max_concurrency=1,
                                  metadata={"chunking": strategy, "chunks": len(index.chunks),
                                            "embeddings": settings.voyage_embed_model,
                                            "reranker": settings.cohere_rerank_model,
                                            "threshold": settings.relevance_threshold})
        rows = _rows(results)
        for row in rows:             # chunk texts are in LangSmith; keep the local copy small
            row["outputs"] = {"hits": [{k: v for k, v in h.items() if k != "text"} for h in row["outputs"]["hits"]]}
        _save_run({"kind": "retrieval", "strategy": strategy, "chunks": len(index.chunks),
                   "experiment": results.experiment_name, "created_at": datetime.now(timezone.utc).isoformat(),
                   "rows": rows})
        print(f"{results.experiment_name}: {strategy} done")


# ---------------------------------------------------------------------------------------------- report
def _latest_runs() -> list[dict]:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS_DIR.glob("*.json"))]
    latest: dict[tuple, dict] = {}
    for r in sorted(runs, key=lambda r: r["created_at"]):
        key = (r["kind"], r.get("strategy"), r.get("suite"), r.get("model"), r.get("temperature"))
        latest[key] = r
    return list(latest.values())


def _bot_seconds(row: dict, run: dict) -> float:
    """Case latency without the rate-limit pauses the runner adds before (re)running a case."""
    retries = (row["outputs"] or {}).get("model_error_retries") or 0
    waits = run.get("pause", 0) + sum(30 * k for k in range(1, retries + 1))
    return max((row["latency_s"] or 0) - waits, 0.0)


def _score(row: dict, key: str) -> float | None:
    return row["checks"].get(key, {}).get("score")


def _stats(client, experiment: str) -> dict:
    """Token and latency totals LangSmith computed for the experiment (best effort)."""
    try:
        p = client.read_project(project_name=experiment, include_stats=True)
        return {"tokens": p.total_tokens, "p50": p.latency_p50.total_seconds() if p.latency_p50 else None,
                "url": p.url}
    except Exception:  # noqa: BLE001 - the report must still be written without LangSmith
        return {}


def _retrieval_section(runs: list[dict]) -> list[str]:
    runs = sorted((r for r in runs if r["kind"] == "retrieval"), key=lambda r: r["strategy"])
    if not runs:
        return []
    out = ["## 1. Retrieval: chunking A vs B", "",
           "Same 47 labelled questions (the PDF test types in English, Spanish and Arabic, plus hidden-style and "
           "must-not-answer questions), same pipeline (BM25 + voyage-4 → RRF → Cohere rerank-v4.0-pro), only the "
           "chunking differs. A fact counts when a top-5 chunk is from the right page and contains the required text. "
           "The \"I don't know\" gate must open for answerable questions and stay shut for the others (threshold 0.76).",
           "", "| Chunking | Chunks | Fact recall@5 | All facts@5 | MRR@10 | \"I don't know\" gate correct | "
               "PDF questions fully passed |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        rows = r["rows"]
        pos = [x for x in rows if _score(x, "fact_recall@5") is not None]
        pdf = [x for x in rows if x["case_id"].startswith("pdf")]
        out.append(f"| {STRATEGY_LABEL.get(r['strategy'], r['strategy'])} | {r['chunks']} | "
                   f"{100 * mean(_score(x, 'fact_recall@5') for x in pos):.1f}% | "
                   f"{100 * mean(_score(x, 'all_facts@5') for x in pos):.1f}% | "
                   f"{mean(_score(x, 'mrr@10') for x in pos):.3f} | "
                   f"{sum(_score(x, 'idk_gate') == 1.0 for x in rows)}/{len(rows)} | "
                   f"{sum(_score(x, 'pass') == 1.0 for x in pdf)}/{len(pdf)} |")
    misses = {r["strategy"]: [x["case_id"] for x in r["rows"] if _score(x, "pass") != 1.0] for r in runs}
    out += [""] + [f"- **{STRATEGY_LABEL.get(s, s)}** misses: {', '.join(m) if m else 'none'}" for s, m in misses.items()]
    return out + [""]


def choose_temperature(sweep: list[dict]) -> dict:
    """Most cases passing every check; then most individual checks passed; then closest to 0.2 (the prior)."""
    def key(run):
        rows = run["rows"]
        cases = sum(_score(x, "pass") == 1.0 for x in rows)
        checks = sum(v["score"] == 1.0 for x in rows for k, v in x["checks"].items() if k != "pass")
        return (-cases, -checks, abs(run["temperature"] - 0.2))
    return sorted(sweep, key=key)[0]


CHECK_COLS = ["status", "prices_grounded", "citations", "facts", "language", "lead_offer"]


def _case_table(run: dict) -> list[str]:
    out = ["| Test | Status | " + " | ".join(CHECK_COLS) + " | Pass |", "|---|---|" + "---|" * (len(CHECK_COLS) + 1)]
    mark = {1.0: "✅", 0.0: "❌", None: "–"}
    for x in run["rows"]:
        out.append(f"| {run['tests'].get(x['case_id'], x['case_id'])} | {(x['outputs'] or {}).get('status', 'error')} | "
                   + " | ".join(mark.get(_score(x, c), "❌") for c in CHECK_COLS)
                   + f" | {mark.get(_score(x, 'pass'), '❌')} |")
    return out


def _answers(run: dict) -> list[str]:
    out = []
    for x in run["rows"]:
        o = x["outputs"] or {}
        fails = [f"{k}: {v['comment']}" for k, v in x["checks"].items() if k != "pass" and v["score"] != 1.0]
        out += [f"### {run['tests'].get(x['case_id'], x['case_id'])}", ""]
        for role, text in x["inputs"].get("history") or []:
            out += [f"> *(earlier {role})* {text[:160]}{'…' if len(text) > 160 else ''}", ">"]
        out += [f"> **User:** {x['inputs']['question']}", "",
                f"<sub>status `{o.get('status')}` · intent `{o.get('intent')}` · language `{o.get('language')}` · "
                f"top rerank score {o.get('top_score') if o.get('top_score') is None else round(o['top_score'], 3)} · "
                f"{_bot_seconds(x, run):.1f}s · search query \"{o.get('search_query', '')}\""
                + (f" · reran {o['model_error_retries']}× after a Gemini 503/429" if o.get("model_error_retries") else "")
                + "</sub>", "",
                o.get("answer") or f"*error: {x['error']}*", ""]
        if fails:
            out += ["**Failed checks:** " + "; ".join(fails), ""]
        out += ["---", ""]
    return out


def write_report(client) -> None:
    runs = _latest_runs()
    pdf_runs = [r for r in runs if r["kind"] == "e2e" and r["suite"] == "pdf"]
    sweep = sorted((r for r in pdf_runs if "lite" not in r["model"]), key=lambda r: r["temperature"])
    extended = [r for r in runs if r["kind"] == "e2e" and r["suite"] == "extended"]
    out = ["# Test results", "",
           f"Generated by `scripts/run_eval.py report` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from the "
           "runs in `eval/results/runs/`. Every run is also a LangSmith experiment (traces of each graph node, router "
           "decision, prompt, token count and latency; the workspace is private, so this file is the shareable copy).",
           "", "Checks are deterministic code (`src/placidway_bot/evaluation/checks.py`), not an LLM judge:", "",
           "- **status**: the bot took the expected path (answered / partial / booking / out of scope / medical / not found)",
           "- **prices_grounded**: every money amount in the reply exists on the pages (or in the user's own question)",
           "- **citations**: answers cite an expected page; refusals cite nothing; never a URL outside the 7 sources",
           "- **facts**: the reply contains the key facts (e.g. `$18,995` and `$30,000`)",
           "- **language**: the reply is in the user's language",
           "- **lead_offer**: booking offers the lead form", ""]
    out += _retrieval_section(runs)
    out += ["## 2. Temperature sweep (answer model)", ""]
    if sweep:
        out += [f"The 8 test questions from the brief through the whole bot. Answer model `{sweep[0]['model']}`, "
                "analyze step cached and identical across runs, so only the answer temperature changes. One run per "
                "temperature (free tier: 20 requests/day), so small differences are within sampling noise.", "",
                "| Temperature | Cases passing every check | Checks passed | Errors | Median latency | Tokens |",
                "|---|---|---|---|---|---|"]
        for r in sweep:
            rows, st = r["rows"], _stats(client, r["experiment"])
            checks = [v["score"] for x in rows for k, v in x["checks"].items() if k != "pass"]
            lat = [_bot_seconds(x, r) for x in rows if x["latency_s"]]
            out.append(f"| {r['temperature']:g} | {sum(_score(x, 'pass') == 1.0 for x in rows)}/{len(rows)} | "
                       f"{sum(c == 1.0 for c in checks)}/{len(checks)} | "
                       f"{sum(1 for x in rows if (x['outputs'] or {}).get('status') == 'error' or x['error'])} | "
                       f"{median(lat):.1f}s | {st.get('tokens', '–')} |")
        best = choose_temperature(sweep)
        out += ["", f"**Chosen temperature: {best['temperature']:g}** (rule: most cases passing every check, then most "
                "checks passed, then closest to the 0.2 prior).", ""]
        headline = [(best, f"answer model `{best['model']}`, temperature {best['temperature']:g}")]
    else:
        out += ["*Pending.* The Gemini 3.5 Flash free tier (5 requests/minute, 20/day; failed 503 \"high demand\" "
                "calls count too) ran out before a valid sweep finished. Run "
                "`python scripts/run_eval.py e2e --suite pdf --temperature <t>` per temperature once the quota resets.", ""]
        headline = []
    headline += [(r, f"answer model `{r['model']}` (the app's fallback model; Flash-Lite ignores temperature)")
                 for r in pdf_runs if "lite" in r["model"]]
    for n, (r, label) in enumerate(headline):
        out += [f"## 3{'abc'[n]}. The 8 test questions — {label}", "",
                f"**{sum(_score(x, 'pass') == 1.0 for x in r['rows'])}/{len(r['rows'])} cases pass every check.** "
                f"LangSmith experiment `{r['experiment']}`. Times are the bot's own; the analyze step is cached in "
                "evaluation, so a live answer takes a few seconds longer.", ""] + _case_table(r) + [""] + _answers(r)
    for r in extended:
        out += [f"## 4. Extended cases — answer model `{r['model']}`", "",
                f"**{sum(_score(x, 'pass') == 1.0 for x in r['rows'])}/{len(r['rows'])} cases pass every check.** "
                f"LangSmith experiment `{r['experiment']}`.", ""] + _case_table(r) + [""] + _answers(r)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"report -> {REPORT.relative_to(ROOT)}")


# ---------------------------------------------------------------------------------------------- CLI
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--strategies", default="recursive,structured")
    e = sub.add_parser("e2e")
    e.add_argument("--suite", choices=["pdf", "extended"], default="pdf")
    e.add_argument("--temperature", type=float, default=None, help="default: GEMINI_TEMPERATURE")
    e.add_argument("--model", default=None, help="answer model, default: GEMINI_MODEL")
    e.add_argument("--pause", type=float, default=13.0, help="seconds between cases (free tier: 5 requests/min)")
    sub.add_parser("report")
    args = parser.parse_args()

    settings = get_settings()
    configure_tracing(settings)
    if not settings.langsmith_api_key:
        sys.exit("LANGSMITH_API_KEY is not set")
    from langsmith import Client
    client = Client()
    if args.cmd == "retrieval":
        run_retrieval(client, args.strategies.split(","))
    elif args.cmd == "e2e":
        temperature = settings.gemini_temperature if args.temperature is None else args.temperature
        run_e2e(client, args.suite, temperature, args.model or settings.gemini_model, args.pause)
    else:
        write_report(client)


if __name__ == "__main__":
    main()
