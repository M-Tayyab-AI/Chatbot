# Write-up: limitations and next steps

## Known limitations

- **Free API tiers.** Gemini 3.5 Flash allows 5 requests per minute and 20 per day, and each question uses two calls.
  When Flash is out of quota or overloaded (503 errors), answers come from Flash-Lite, the fallback model. Voyage
  (3 requests per minute) and Cohere's trial key (10 per minute, about 1,000 per month, and not licensed for
  production) limit the bot to roughly 3 questions per minute.
- **The end-to-end test results come from Flash-Lite.** Flash's daily quota ran out during evaluation, partly because
  failed 503 calls also count against it. The temperature sweep therefore hasn't run. Flash-Lite ignores temperature,
  so these results say nothing about it.
- **Seven pages only.** Anything outside them, such as dental implants in Mexico or other clinics, gets an honest
  "not in our content" reply and a free-quote link. Video content is limited to titles and descriptions, with no
  transcripts.
- **Refresh on Streamlit Cloud is temporary.** The admin button rebuilds the index on the app's disk, which resets on
  restart. A permanent refresh means running `scripts/ingest.py` and pushing the result.
- **The checks don't catch everything.** Every price and citation is verified in code. The wording of an answer is
  not: a sentence can be phrased loosely as long as its numbers and sources are right. Flash-Lite also sometimes
  labels a complete answer as partial, which adds an unnecessary "request a quote" line.
- **Latency.** An answer takes about 3–8 seconds, and a sleeping app needs about a minute to wake up.

## With another week

1. **Finish the model evaluation.** Run the 8 tests and the temperature sweep on Gemini 3.5 Flash. Grow the test set
   to about 100 questions in the style of the hidden ones, and add an LLM judge for answer faithfulness alongside the
   existing deterministic checks.
2. **Answer price questions from structured data.** Route price questions to a lookup over the extracted price
   records instead of free-text generation. The price would then be copied exactly from the record, and the LLM would
   only write the sentence around it.
3. **Automatic, permanent refresh.** Run a nightly GitHub Action that re-crawls the pages, re-indexes only those
   that changed, and commits the result, which redeploys the app. Move the vectors into Neon (pgvector) so refreshes
   survive restarts, and report pages whose content changed.
4. **Production readiness.**
   - Paid Gemini and Cohere keys
   - Streamed answers, so the reply appears as it is written
   - Thumbs-up/down feedback on each answer
   - A weekly review of the unanswered-questions log to find content gaps
5. **More coverage.** Add video transcripts, and follow the compare page's pagination if PlacidWay allows it.
