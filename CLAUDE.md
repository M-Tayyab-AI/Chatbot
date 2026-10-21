# CLAUDE.md — PlacidWay Website Knowledge Chatbot

Assessment for the **AI Automation Engineer** role at PlacidWay (medical tourism). Source brief:
`AI Automation Assessment Website Knowledge Chatbot.pdf` (repo root).

**Goal:** a small, reliable chatbot that answers visitor questions **only** from 7 PlacidWay pages, cites the
source URL(s) on every answer, says "I don't know" honestly, and never invents prices, clinics or medical facts.
Accuracy + honesty = 50% of the rubric. A polished UI with one invented price fails.

> Every decision below was confirmed with the user. Do not change a decision without asking first.
> Do not start building a new phase until the user says so.

---

## 1. Source URLs (the ONLY knowledge base — never crawl beyond these)

| # | page_type      | URL |
|---|----------------|-----|
| 1 | clinic_profile | https://www.placidway.com/profile/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center |
| 2 | package        | https://www.placidway.com/package/7903/Alternative-Adenocarcinoma-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC |
| 3 | package        | https://www.placidway.com/package/7902/Alternative-Adrenal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC |
| 4 | package        | https://www.placidway.com/package/7901/Alternative-Anal-Cancer-Treatment-Package-in-Tijuana-Mexico-by-ITC |
| 5 | clinic_videos  | https://www.placidway.com/profile-videos/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center |
| 6 | price_list     | https://www.placidway.com/price-list/5/Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center |
| 7 | price_compare  | https://www.placidway.com/search-medical-pricings/Alternative-Medicine/All/1 |

These live in `config/sources.yaml` (single source of truth). Fallback CTA links: free quote / contact →
`https://www.placidway.com/contact-us` (found on the pages; confirm the dedicated quote URL during build).

**Crawl etiquette (required by the brief):** obey robots.txt (checked: all 7 paths allowed; `/ajax/`, `/api/v1/`,
`/uploads/` disallowed), clear custom User-Agent (configured via `CRAWLER_USER_AGENT` env — do not hard-code the
user's personal email), ~1 request/second, retries with backoff.

---

## 2. Confirmed tech decisions

| Area | Decision |
|------|----------|
| Language | Python 3.12 (`.venv` via `python -m venv`) |
| Ingestion | **DECIDED (tested): `requests` + BeautifulSoup with the `html5lib` parser.** All 3 methods were built and scored (`docs/scraper_comparison.md`): requests = 100% visible-content coverage, 0 leaks, deterministic hashes; Selenium = site auto-translates by geo-IP into Urdu (5/7 pages 29–88% translated, broken price ranges) → **Selenium code + dependency deleted** (user request). Claude-in-Chrome capture kept only as the verified reference copy in `data/raw/chrome/`. |
| Storage | **No PDF.** Per page: `data/raw/<method>/<slug>.html` (archive only, never chunked) → `data/processed/<slug>.json` (clean structured, plain text) + `<slug>.md` (human-readable). Test asserts **zero HTML tags / boilerplate** in processed output. |
| Chunk metadata | `{chunk_id, source_url, page_title (source name), page_type, section, fetched_at, content_hash}` — section replaces "PDF page number" as the citation anchor. |
| Chunking | **DECIDED (tested): structured** — one chunk per heading section (tiny siblings merged, >1500 chars split), FAQ Q+A = one chunk, each price record = atomic chunk, videos in groups of 4, every chunk prefixed with `[page type: treatment/clinic \| clinic \| location]` + page title + section path. Recursive 1000/200 lost on the template-confusion questions (see `docs/chunking_comparison.md`). Production index: 219 chunks. |
| Embeddings | **Voyage `voyage-4`** (multilingual, 1024 dims) |
| Reranker | **DECIDED (user): Cohere `rerank-v4.0-pro`** (`RERANKER=cohere`), Voyage rerank-3 kept as switch. Rerank pool = fused top-20 + vector top-10 safety net. **"I don't know" threshold = 0.76** (Cohere Pro; Voyage would be ~0.48). Result: 27/27 PDF tests in EN/ES/AR, 100% fact recall on 47 questions. |
| Vector store | **Chroma** (local persistent, `data/index/`) + **BM25** (`rank_bm25`) → hybrid retrieval → rerank |
| LLM | **DECIDED (user, free tier): split models** — analyze (rewrite+route) on `gemini-3.5-flash-lite` (`GEMINI_ANALYZE_MODEL`), answer generation on `gemini-3.5-flash` (`GEMINI_MODEL`). Each has its own free-tier daily quota (3.5 Flash = 20 requests/day). While 3.5 Flash quota is exhausted, test with `GEMINI_MODEL=gemini-3.5-flash-lite` (env override only). Lite ignores temperature (fixed sampling). Thinking level `low`. |
| Temperature | Sweep **0.1 / 0.2 / 0.3 / 0.5 / 1.0(default)** on the eval set, pick the best, record results. Note: Google advises not tuning temperature for Gemini 3.x — we test it rather than assume. User's prior: 0.2. |
| Orchestration | **LangGraph** (LangChain components inside: `langchain-google-genai`, retrievers) |
| UI + deploy | **Streamlit** → Streamlit Community Cloud from a **public GitHub repo**. All logic in `src/` functions; Streamlit only calls one entry function (e.g. `answer(question, session_id)`) that returns answer + citations. |
| Persistence | **Neon Postgres (free)** — chat history (LangGraph Postgres checkpointer), unanswered-question log, leads. Streamlit Cloud FS is ephemeral. |
| Memory | Per-session history, `session_id` kept in URL query param so reload restores the chat; full history persisted; **last 10 messages** sent to the LLM; rewrite node makes follow-ups standalone. |
| Refresh | **A) CLI** `python scripts/ingest.py` + **B) admin "Refresh now" button** in Streamlit sidebar (protected by `ADMIN_TOKEN`). Both call the same function: re-fetch → content-hash diff → re-parse/re-chunk/re-embed **only changed pages** → atomic index swap → update `manifest.json`. No scheduled job. |
| Keep-awake | **None.** User wakes the app from Streamlit dashboard. README tells reviewers a sleeping app wakes with one click. |
| Eval | **LangSmith** (tracing of every LLM call, graph node, router decision, tokens, latency; experiments comparing chunking A vs B and temperatures) + **pytest** (deterministic unit tests) + exported `eval/results/test_results.md` for submission. |
| Secrets | Only in `.env` (gitignored) / Streamlit secrets. `.env.example` lists every key. Never commit secrets. |
| Git | `git init` locally; user creates the public GitHub repo. |

### Nice-to-haves (ALL in scope)
1. Answer in the user's language (Spanish/Arabic/Urdu → retrieve in English, reply in user's language, prices/URLs verbatim).
2. Follow-up questions via chat history ("what about in Turkey?").
3. Short medical disclaimer **once per session** when a medical topic first appears.
4. Lead capture: on quote/booking intent, offer name + email form (validated) → Neon; quote link as fallback.
5. Log unanswered / out-of-scope / blocked questions → Neon (question, reason, top retrieval score, timestamp); visible in admin sidebar.

---

## 3. Architecture (LangGraph)

```
question + last 10 msgs
  → rewrite      (standalone, English search query; keep user language for the reply)
  → route        (cost | comparison | procedure | general | out_of_scope | medical_advice | booking | lead | chitchat)
  → retrieve     (hybrid BM25 + Chroma, metadata filters by page_type) → rerank-3 → relevance threshold
  → generate     (Gemini, strict JSON: {answer, cited_chunk_ids, found, needs_disclaimer})
  → validate     (cited ids exist; every price/currency in answer appears verbatim in cited chunks;
                  citations map to the 7 allowed URLs; wrong-premise check)
  → fallback     ("not in our content" + free-quote/contact link) on any failure / low score
  → log          (unanswered → Neon) ; lead → Neon
```

Guardrail rules the prompt + validator must enforce:
- Prices always with currency + page context ("starting from", Out-Patient vs In-Patient, includes/excludes).
- Page has no price → say "price not listed" + suggest free quote.
- Comparison only if **both** sides exist in content **and** refer to the same procedure.
- Never agree with a user-supplied price that isn't in the content (test #8 "$50").
- No diagnosis / safety judgement for a personal condition → recommend a doctor (test #7).
- Booking → can't book; point to contact/quote form (test #5). Weather etc. → polite decline (test #6).

---

## 4. Findings from page inspection (2026-10-20) — parser must handle these

- **Server-rendered HTML**; prices present without JS. Main content = `main.article-dt-desc` → `section.js-toc-section[id]`.
- **Strip** before extraction: popups (`#book_consultation_popup`, `#request_video_call_popup`, `#confirm_vs_popup`),
  sidebar (`.col-md-4`, `#cta-form`), stats widget, `.related-widgets-row` / related sliders (**contain OTHER clinics'
  prices** — e.g. $24,000, $10,000, $4,500, $33,400 — leaking them = invented prices), nav, language list,
  `<script>` inside main (JSON-LD text is inside `main` on the compare page → duplicated FAQ text).
- **JSON-LD is valuable structured truth:** profile has `FAQPage`; price list has 7 `Product` offers (USD 18995);
  packages have `Product`; videos page has **124 `VideoObject`** (name, short description, YouTube embed, no transcript).
- **Profile** sections: about(~10K chars, FAQ x7, table "Base Package Price $18,995 / cancer types / inclusions"),
  details, treatments, qualifications (4 doctors), testimonials, awards, destination (Tijuana travel), reviews.
- **3 packages share one template** (risk: retrieval confusing adrenal/anal/adenocarcinoma → chunk headers must name
  the treatment): table Out-Patient 3 weeks **$18,995 USD**, In-Patient 3 weeks **$30,000 USD**; "Included in…"
  lists; Optional Specialized Procedures (additional cost: $2,600, $3,950, $4,250, $3,800, $4,650, $6,200, $7,200,
  $200, $462, $950, $300); Excluded Services; 3-week itinerary; 8 FAQs each.
- **Price list**: 7 rows all `$18995` USD; one row has **empty name in HTML** → JSON-LD says "Cancer Treatment";
  "Colorectal Cancer" row has details "18995 USD". Use JSON-LD to fill gaps.
- **Price compare** (`/All/1` only — do not follow pagination or category links): generic cost-range table
  (Acupuncture $200–$800, Panchakarma 14d $800–$2,500, Ozone $500–$1,500, Alt. Stem Cell $3,000–$10,000,
  Naturopathic retreat $1,200–$4,000), component table (consultation $50–$250 …), clinic table:
  Thailand Bangkok $64 (Physio/Massage), Croatia Murter $70 (Physio/Massage), **Turkey Konya $110 (Ozone)**,
  **Mexico Guadalajara $1000 (Ozone)**, Mexico City/Puebla $5500 (Live Cell). Plus included/not-included lists, FAQ.
- **No explicit "recovery time"** anywhere → test #4 follow-up must say it's not listed (may mention the 3-week
  program and post-treatment follow-up support that *are* on the page) — never invent.
- "Dental implant in Mexico" (brief's example) is **not** in this content → honest "I don't know".

### Found while building ingestion (Phase 1)
- **Use `html5lib`**: pages have unbalanced tags. `lxml` nests the sidebar (quote form, site stats) inside `<main>`;
  `html.parser` doesn't auto-close `<p>` before headings (lost 19 sections per package).
- **Site JS machine-translates by geo-IP** (gtranslate) → never use a headless browser; requests gets original English.
- **Review cards** render text *before* the reviewer name → flattened to "Patient review by NAME (DATE, 5/5): …".
- **Related links inside packages are randomised per load** (e.g. "All on 6 Dental Implants in Tijuana") → stripped
  (would also break hash stability and fake an answer for the dental question).
- **"About Medical Center" aside** (outside main) = this clinic's facts (address Blvd. Padre Kino 10133, languages
  EN/ES/AR/FR) → extracted deliberately. Its giant "all videos" link list is dropped.
- **Compare page "Load More"** = client-side over inline `center_data` JSON (same 5 rows, no AJAX). Listing notes
  in that JSON are not visible but essential: Turkey $110 = 1 session (10 sessions $1000); Mexico $1000 = 5 sessions;
  VISANI "starting at 5,500 USD". `INCLUDE_LISTING_NOTES = True` in `parsers/pages.py` (pending user confirmation).
- Compare-page FAQ answers are in collapsed `<details>` (invisible in a browser text copy, captured by requests).
- Content hashes are stable across fetches (7/7) → refresh only re-embeds changed pages.

### Rate limits (MVP stays on free/trial tiers — user decision)
- Voyage free tier: 3 requests/min AND 10K tokens/min. Cohere trial: 10 rerank calls/min, ~1,000/month, "not for
  production" per Cohere ToS → note in README. Per question: 1 Voyage call (query embed) + 1 Cohere call → ~3 q/min.
- Everything is cached (SQLite `data/index/voyage_cache.sqlite`, shared by both providers).

### Gemini free tier = 20 requests/DAY per model (found 2026-10-21, Phase 3 live run)
- Error: `generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash` (the `gemini-3.5-flash` id is
  served/quota'd as 3.6). Bot uses 2 Gemini calls per turn (analyze + answer) → ~10 questions/day. Blocks the live
  test, the LangSmith eval/temperature sweep and the 7-day review. User chose to stay free: split models (see LLM).

### Phase 4 (Streamlit + Neon) — built 2026-10-21
- `streamlit_app.py` (UI only) → `ChatBot` (`src/placidway_bot/chat.py`). Neon: LangGraph `PostgresSaver`
  (pool with `prepare_threshold=0`, `autocommit`), tables `leads`, `unanswered_questions` (`services/db.py`).
  No DATABASE_URL / Neon down → in-memory history + JSONL logs (app keeps working).
- Session id in URL `?sid=` → reload restores chat from Neon (verified in browser). Last 10 messages to the LLM.
- Lead form on booking/quote intents + sidebar "Free quote"; validated (`services/leads.py`). Admin expander
  (ADMIN_TOKEN): Refresh now (pipeline.refresh + reload index), page status, unanswered log, leads.
- Gotchas fixed: Streamlit runs turns in worker threads → SQLite caches use `check_same_thread=False` + lock;
  Streamlit renders `$…$` as LaTeX → UI escapes `$` (`md()`), otherwise prices are garbled.
- Answer model falls back to `GEMINI_FALLBACK_MODEL` (Lite) via `with_fallbacks` when 3.5 Flash fails (daily quota).

### Phase 5 (LangSmith eval) — started 2026-10-21
- `scripts/run_eval.py` (retrieval | e2e | report) + `eval/test_questions.yaml` (suites `pdf`, `extended`) +
  deterministic checks `src/placidway_bot/evaluation/checks.py`. LangSmith datasets: placidway-retrieval,
  placidway-pdf-tests, placidway-extended-tests. Local copies in `eval/results/runs/`, report `eval/results/test_results.md`.
- Retrieval A vs B done: recursive 80.6% fact recall / 20 of 27 PDF; structured 100% / 27 of 27.
- Bug fixed: `configure_tracing` now exports LANGSMITH_API_KEY/PROJECT/ENDPOINT (pydantic-settings doesn't set env).
- Gemini 3.5 Flash free tier: 5 requests/MIN and 20/DAY; **503 "high demand" failures and LangChain auto-retries
  also count toward the daily 20** — the first sweep attempt burned the day's quota without one valid run.
  Quota resets at midnight Pacific (12:00 Pakistan time).
- User chose **C**: full e2e run on Flash-Lite (fallback model) — pdf 7/8, extended 6/7, zero invented prices,
  zero wrong citations. Fails: t3 labelled `partial` (correct content, extra CTA); Arabic medical question answered
  as grounded "pages don't say + see your doctor + disclaimer" instead of the template (expectation too narrow).
  Flash temperature sweep still pending (needs quota: option A wait / B billing).
- Google docs: **3.5 Flash-Lite and 3.6 Flash ignore temperature/top_p/top_k** (fixed 1.0/0.95/64, no error);
  **3.5 Flash supports temperature 0–2**. Our `gemini-3.5-flash` quota errors name `gemini-3.6-flash` → before the
  sweep, check the response's model version with 1 call; if it is 3.6, the sweep measures nothing → tell the user.

### Phase 3 (LangGraph) learnings
- Lite once routed "¿Cuánto cuesta…?" as quote_request → quote_request now ALSO retrieves (answers + lead form),
  so a misroute can't swallow an answer. Prompt says cost questions are always `knowledge`.
- Partial answers (e.g. recovery time not on page) must be answer_found=true + fully_answered=false, not "not found".
- `medical_topic` only for health/safety/suitability (not prices) → disclaimer once per session where relevant.
- Live run (Lite for both, `data/experiments/demo_run_lite.md`): 8/8 PDF tests + follow-up "What about in Turkey?"
  + Spanish + Arabic + dental-implant "not found" all behave as expected.

### Voyage free tier details
- Confirmed by API error. User chose to stay on the free tier for the MVP. `VoyageService` has SQLite cache + backoff; set
  `VOYAGE_MIN_INTERVAL_SECONDS=30` for batch jobs on the free tier.

---

## 5. File structure

```
placidway-chatbot/
├── CLAUDE.md  README.md  .env.example  .gitignore  requirements.txt
├── streamlit_app.py                 # thin UI: calls src entry function only
├── config/sources.yaml
├── data/
│   ├── raw/{requests,selenium,chrome}/
│   ├── processed/                   # <slug>.json + <slug>.md (committed)
│   └── index/                       # chroma, bm25, manifest.json (committed for Streamlit Cloud)
├── src/placidway_bot/
│   ├── config.py                    # pydantic-settings from env
│   ├── ingest/  fetcher.py  scrapers/requests_bs4.py  models.py  render.py
│   │            parsers/{base,pages}.py   # generic HTML->sections + page-type rules
│   │            chunker.py  pipeline.py (refresh)
│   ├── retrieval/  voyage_service.py (cache+backoff)  cohere_reranker.py  store.py (Chroma+BM25)
│   │               hybrid.py (RRF + rerank pool)  retriever.py (KnowledgeRetriever: search -> hits + confident)
│   ├── graph/  state.py  nodes.py  prompts.py  templates.py (fixed replies en/es/ar/ur)  build.py
│   ├── guardrails/  price_check.py  citations.py
│   ├── services/  db.py  leads.py  gap_log.py
│   └── chat.py                      # answer(question, session_id) entry point
├── scripts/  ingest.py  fetch_raw.py  parse_raw.py  compare_scrapers.py  compare_chunking.py  run_eval.py
├── data/experiments/                # per-method parsed output + experiment indexes + scores (comparison evidence)
├── eval/  retrieval_labels.yaml  test_questions.yaml  results/
├── tests/
└── docs/  scraper_comparison.md  chunking_comparison.md  writeup.md
```

## 6. Deliverables checklist (from the brief)
- [ ] Code in public GitHub repo
- [ ] README: setup, architecture, tools & models + why
- [ ] Test results: answers to the 8 test questions (`eval/results/test_results.md`)
- [ ] Half-page write-up: improvements with another week + known limitations (`docs/writeup.md`)
- [ ] Live Streamlit URL, running ≥ 7 days after submission

## 7. Working rules
- Ask the user before each new phase / any decision not listed here.
- Code must be readable, modular, typed, with error handling — reviewer will ask for a live change in a 45-min call.
- Run commands from the venv: `.venv\Scripts\python`.
