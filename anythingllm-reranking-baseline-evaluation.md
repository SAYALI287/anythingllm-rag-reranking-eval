# AnythingLLM Retrieval Evaluation: Default vs. Accuracy Optimized (Reranking) Mode

**Test document:** *The Immortals of Meluha* by Amish Tripathi (uploaded as a workspace document)
**Model:** llama3.2:3b (via Ollama, local)
**Embedding model:** nomic-embed-text
**Vector DB:** LanceDB
**Date tested:** July 9, 2026

## Background

AnythingLLM ships with a built-in, fully-implemented reranking feature (`vectorSearchMode: "rerank"`, labeled "Accuracy Optimized" in the workspace UI), sitting alongside the default vector-search-only mode. This document records a baseline comparison between the two modes on 9 questions, before any code changes were made, to establish what problem (if any) is worth solving.

## Methodology

- Uploaded the same document to a single workspace.
- Asked each question in **Default mode** first, then switched the workspace's Vector Search Mode to **Accuracy Optimized** and asked the identical question in a fresh thread (to avoid conversation-memory bleed-through).
- Recorded the full answer text and the response time/token-speed shown by AnythingLLM for each answer where available.
- Verified one specific factual claim (Shiva's origin) against external sources (Wikipedia, published book reviews) to check whether "more confident" answers were actually "more correct."

## Results

| # | Question | Default Answer (summary) | Reranked Answer (summary) | Default Time | Reranked Time | Verdict |
|---|---|---|---|---|---|---|
| 1 | What is this story about? | Vague; calls contexts "unrelated"; hedges heavily | Specific, coherent, names book/author/character correctly | 10.5s (16.26 tok/s) | 5.4s (17.84 tok/s) | **Reranked better** (more accurate, also faster) |
| 2 | What happened when the storm hit? | Self-corrects a hallucinated detail from Q1 (first pass: 2.2s said no storm mentioned; second pass: 5.2s found and explained it) | Self-corrects a similar hallucinated detail from Q1 | 2.2s / 5.2s (18.85 / 17.16 tok/s) | 5.9s (16.94 tok/s) | Tie (shared failure); default faster on this one |
| 3 | Who is Nandi? | Reasonably accurate, calls him a companion/advisor | More detailed, adds mythological background | 5.5s (16.72 tok/s) | 9.0s (15.40 tok/s) | Reranked more detailed, but slower here |
| 4 | What are the main themes? | Reasonable, thoughtful list | Reasonable, thoughtful, differently-emphasized list | 10.1s (15.00 tok/s) | 15.2s (14.63 tok/s) | Tie on quality; **default faster** |
| 5 | Where does Shiva come from? | Honest hedge: "shrouded in mystery," admits uncertainty | **Confidently wrong: "Shiva comes from Kashmir"** (fact-checked: actually Mount Kailash, Tibet) | 11.1s | 0.33s | **Default better (reranked hallucinated)**, reranked much faster |
| 6 | Why does Shiva agree to go to Meluha? | Grounded, cites a specific context, describes protective motive | Contradicts default — claims no such agreement is mentioned, describes reluctance instead | 6.8s | 2.6s | Disagreement — unresolved, flagged for follow-up |
| 7 | What is the difference between Shiva and Nandi? | Specific, includes a quote-like detail ("my Lord" / "my friend" exchange) | Vaguer, falls back on generic mythology knowledge | 10.4s | 2.4s | Default better, reranked faster |
| 8 | What does Shiva do for a living before the story begins? | Honest: "text doesn't provide this" | Honest: "no information provided" | 7.8s | 1.2s | Tie (both correctly decline to answer) |
| 9 | What does Shiva think about smartphones? (trap question) | Correctly identifies as anachronistic, refuses to hallucinate | Correctly identifies as anachronistic, refuses to hallucinate | — | — | Tie (both pass) |

**Note on timing:** the speed advantage of reranked mode was not universal. On questions 1, 5, 7, and 8 reranked mode was substantially faster (2x–30x). On questions 2, 3, and 4, however, reranked mode was actually *slower* than default. This means the earlier "reranking is always faster" impression from a smaller sample was too strong a claim — the real pattern is mixed and likely depends on how many candidate chunks the reranker pulls in for a given query (recall the code's dynamic `searchLimit`, which scales with total embeddings in the workspace and could vary in overhead per query).

## Headline Findings

### 1. Reranked mode's speed advantage is real but inconsistent, not universal
On 4 of 7 timed questions, reranked mode was substantially faster than default (2x–30x, e.g. 11.1s → 0.33s on Q5). On the remaining 3, it was actually slower (e.g. Q4: 10.1s default vs. 15.2s reranked). This means the common assumption that "reranking adds latency" doesn't hold uniformly here, but neither does the opposite claim that it's always faster — the effect appears query-dependent, plausibly tied to how many candidate chunks the reranker pulls in for that specific query before narrowing down (the code's `searchLimit` scales dynamically with total embeddings in the workspace).

### 2. Reranked mode is not more accurate — and can be confidently wrong
On the one question where a factual claim could be independently verified (Q5, Shiva's origin), reranked mode produced a short, confident, **incorrect** answer ("Kashmir"), while default mode gave a longer, hedging, but ultimately more honest answer (admitting the origin wasn't clear from the retrieved context). This is the most important finding of this evaluation: **narrower, more "precise" retrieval does not guarantee more truthful output — it can instead produce shorter, more confident, and more wrong answers, with less visible hedging to signal uncertainty to the user.**

### 3. Default mode tends to be more verbose and more likely to hedge
Across most questions, default mode's answers were longer and more likely to include phrases acknowledging uncertainty ("it seems," "difficult to say without more context"). Reranked mode's answers were shorter and more declarative. This is a plausible side effect of context size: default mode retrieves 4 chunks by raw similarity (which may be more topically diverse/redundant), giving the LLM more material to hedge and elaborate around; reranked mode retrieves a smaller, tightly-focused set of the *most* relevant chunks from a larger candidate pool, giving the LLM less raw material — leading to shorter, more decisive (but not necessarily more correct) answers.

## Conclusion and Next Step

Reranking, as currently implemented in AnythingLLM, offers a clear and substantial **speed** benefit but does **not** demonstrate a clear **accuracy** benefit in this test, and shows a real risk of producing confidently incorrect answers with less hedging than the default mode. This is a legitimate, evidence-backed problem worth addressing in code, rather than a purely cosmetic or theoretical one.

**Planned code change:** modify the reranked retrieval path to reduce overconfident hallucination while preserving most of the speed advantage. Candidate approaches under consideration:
- Widen `topN` slightly in reranked mode (e.g. 4 → 6) to restore some of default mode's contextual breadth.
- Add a grounding instruction to the prompt used in reranked mode, explicitly asking the model to express uncertainty when retrieved context doesn't clearly support a claim.
- Blend in 1-2 "diversity" chunks from the wider candidate pool (not just the top-N by rerank score) alongside the top reranked chunks.

The same 9 questions above will be re-run against the modified implementation to measure whether the change reduces confident-hallucination cases (like Q5) without meaningfully reversing the speed advantage.

## Implementation

**Change made:** in `server/utils/chats/stream.js`, added a conditional grounding instruction appended to the system prompt, active only when `workspace.vectorSearchMode === "rerank"`:

```javascript
const groundingInstruction =
  workspace?.vectorSearchMode === "rerank"
    ? "\n\nIMPORTANT: The context above was selected by a precision-focused retrieval process that returns a small, narrow set of passages. If the provided context does not clearly and directly support an answer, explicitly say you are not certain rather than guessing or stating an answer with unwarranted confidence."
    : "";

const finalSystemPrompt = systemPrompt + groundingInstruction;
```

`finalSystemPrompt` (instead of the original `systemPrompt`) is then passed into `LLMConnector.compressMessages(...)`. This is a minimal, contained change: it does not touch retrieval logic, the reranker itself, or default (non-reranked) mode behavior at all — the instruction is an empty string unless reranking is active.

## Results After Fix (Reranked mode, with grounding instruction)

| # | Question | Before Fix (reranked) | After Fix (reranked) | Change |
|---|---|---|---|---|
| 1 | Where does Shiva come from? | Confidently wrong: "Shiva comes from Kashmir" | "I am not certain... does not clearly state where Shiva comes from" (3.3s) | ✅ **Fixed** — the target case |
| 2 | What is this story about? | Confident, accurate — correctly named book/author/protagonist | Now hedges heavily; gives a confusing, seemingly unrelated answer (5.5s) | ❌ **Regression** — over-hedged a previously-good answer |
| 3 | What happened when the storm hit? | Both modes previously self-corrected a hallucinated detail | Now gives a fully confident, detailed (and still unverified/likely hallucinated) account of a boat capsizing (7.5s) | ❌ **Not caught** — fix did not apply here |
| 4 | Who is Nandi? | More detailed, specific mythological background | Now hedges, calls itself "not certain" despite reasonable underlying info (4.2s) | ⚠️ Over-hedged |
| 5 | What are the main themes? | Reasonable, specific list | Now hedges, notes the context is mostly "reviews and praise" rather than committing to an answer (5.5s) | ⚠️ Over-hedged |
| 6 | Why does Shiva agree to go to Meluha? | Contradicted default mode; described reluctance instead of agreement | Now explicitly says it's not certain, describing the ambiguity honestly (5.0s) | ✅ Improved (surfaces the ambiguity instead of asserting one version) |
| 7 | Shiva vs Nandi difference | Vague, fell back on generic mythology knowledge | Now explicitly flags insufficient context rather than guessing (3.7s) | ➖ Similar outcome, now explicit about the gap |
| 8 | Shiva's job before the story | Honest "no information provided" | Still honest, now also mentions a context detail (chief scientist responsibility) not surfaced before (3.7s) | ➖ Similar, no meaningful change |
| 9 | Smartphones (trap question) | Correctly refused to hallucinate | Still correctly refuses, with hedging language stylistically consistent with the rest (4.0s) | ➖ No meaningful change |

## Honest Assessment of the Fix

**What worked:** the grounding instruction directly and successfully corrected the specific failure case that motivated it (Q1/Q5) — the model now expresses genuine uncertainty instead of stating a fabricated fact with confidence.

**What didn't work / new problem introduced:** the instruction is a blanket addition to every reranked response, not a targeted response to actual retrieval confidence. As a result, it **over-hedged previously-correct, well-grounded answers** (Q2, Q4, Q5) — answers that were accurate before the fix became vaguer and less useful after it. It also **failed to catch a known separate hallucination** (Q3, the storm/boat scene), showing the fix does not reliably distinguish well-grounded from poorly-grounded responses — it simply nudges the model's general tone toward hedging, somewhat inconsistently.

**Conclusion:** a static, always-on prompt instruction is a partial, blunt-instrument mitigation, not a complete fix. It trades some hallucination risk for a measurable increase in unnecessary hedging on answers the model actually had good grounding for. A more robust follow-up (not implemented in this pass) would condition the grounding instruction — or its strength — on an actual retrieval confidence signal (e.g. the reranker's own relevance/similarity scores from `rerankedSimilarityResponse`) rather than applying it uniformly whenever reranking is simply turned on. This is a natural next iteration rather than a finished solution.

## Follow-up Iteration: Confidence-Based Grounding

Based on the limitation identified above, a second version was implemented: instead of firing the grounding instruction on every reranked response, it now fires only when the reranker's own average relevance score (`avgRerankScore`, computed from the individual `rerank_score` values already produced inside `rerankedSimilarityResponse`) falls below a threshold (initially set to `0.5`).

**Implementation:** `avgRerankScore` is computed at the end of `rerankedSimilarityResponse` in `server/utils/vectorDbProviders/lance/index.js`, threaded through `performSimilaritySearch`'s return value, and checked in `server/utils/chats/stream.js`:

```javascript
const RERANK_CONFIDENCE_THRESHOLD = 0.5;
const isLowConfidenceRerank =
  workspace?.vectorSearchMode === "rerank" &&
  typeof vectorSearchResults.avgRerankScore === "number" &&
  vectorSearchResults.avgRerankScore < RERANK_CONFIDENCE_THRESHOLD;

const groundingInstruction = isLowConfidenceRerank
  ? "\n\nIMPORTANT: The retrieved context for this question had a low average relevance score, meaning it may not fully support a confident answer. If the provided context does not clearly and directly support an answer, explicitly say you are not certain rather than guessing or stating an answer with unwarranted confidence."
  : "";
```

### Real observed scores (from live debug logging)

| Question | avgRerankScore | Triggered? | Actual model behavior |
|---|---|---|---|
| Where does Shiva come from? | 0.632 | No | Answered without hedging (no hallucination this run) |
| What is this story about? | 0.0012 | Yes | Hedged correctly ("I am not certain...") |
| What happened when the storm hit? | 0.0028 | Yes | **Still confidently hallucinated the fabricated boat/storm scene**, despite the instruction being active |

### Two deeper problems this surfaced

**1. The LLM does not reliably follow the grounding instruction even when it is present in the prompt.** On the storm question, the instruction fired (confirmed via debug log) but the model still generated a detailed, unhedged, fabricated account. This indicates that a text-based instruction is not a reliable control mechanism on its own, at least for a small local model (`llama3.2:3b`) — the model can and does ignore it when the retrieved (but irrelevant) context reads as a narratively coherent scene.

**2. The rerank score scale is not obviously trustworthy or comparable across queries.** The "storm" and "story" questions scored near zero (0.001–0.003), while the "Shiva's origin" question scored 0.63 — a gap large enough to suggest the underlying `rerank_score` values from the cross-encoder are not naturally normalized to a clean, comparable 0–1 range (cross-encoders often output raw, unbounded logits rather than calibrated probabilities). This undermines confidence that a single fixed threshold (0.5, chosen without prior calibration) is a meaningful or stable cutoff, and raises the question of whether "average of individual chunk scores" is even the right aggregate statistic to use.

### Honest conclusion on this iteration

The confidence-based approach is conceptually the right direction — targeting the instruction at genuinely weak retrieval rather than applying it blindly — but this implementation is not yet reliable. It did successfully avoid hedging on the one previously-good, high-scoring answer tested here, which is an improvement over the blanket version. However, it did not solve the harder underlying problem: the model's willingness to state a plausible-sounding but ungrounded narrative with full confidence, regardless of whether it was told to be cautious. A more complete solution would likely need either (a) a validated, calibrated confidence score (e.g. via score normalization or a held-out calibration set) rather than a guessed threshold, and/or (b) a mechanism stronger than a prompt instruction — such as filtering out low-relevance chunks entirely before they reach the model, rather than asking the model to self-police its use of them.

