# AnythingLLM: Evaluating and Improving the Built-in RAG Reranking Feature

A hands-on investigation into [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)'s existing but under-documented "Accuracy Optimized" (reranking) retrieval mode — measuring its real-world impact on answer accuracy and speed, discovering a confident-hallucination failure case, shipping a targeted fix, and honestly reporting the fix's limitations.

## TL;DR

- AnythingLLM has a fully-implemented, two-stage retrieve-then-rerank RAG pipeline built in, exposed as a per-workspace "Accuracy Optimized" setting — but it's easy to miss and its real-world tradeoffs aren't documented anywhere.
- I ran a 9-question evaluation comparing default vector search against reranked search on a real document, with response timing and an external fact-check.
- **Finding:** reranking's speed advantage is real but inconsistent (2×–30× faster on some queries, slower on others), and it is **not** more accurate — it produced one confidently wrong answer that default mode correctly hedged on.
- I traced the exact code path (`server/utils/chats/stream.js` → `server/utils/vectorDbProviders/lance/index.js`) and added a conditional grounding instruction to the system prompt, active only in reranked mode.
- **Result:** the fix corrected the target hallucination, but also caused the model to over-hedge on previously-correct answers, and it did not catch a separate, known hallucination case. Documented as a partial mitigation, not a complete fix — with a specific, evidence-based direction for the next iteration.

## Why this project

Most "add RAG re-ranking" tutorials build the feature from scratch. AnythingLLM already had it — fully implemented, benchmarked in a code comment by the original developer, and completely unused by default. That made for a more realistic engineering exercise: read an unfamiliar codebase, find out whether an existing feature actually does what it claims, and fix a real problem it surfaced, rather than build a feature in a vacuum.

## What I actually did

1. **Environment setup** — ran AnythingLLM via Docker, then switched to running it from source in dev mode (Node.js + Ollama for a fully local, free LLM/embedding stack) to be able to read and edit the real code.
2. **Codebase exploration** — traced the full retrieval pipeline: `stream.js` (chat orchestration) → `performSimilaritySearch` → `similarityResponse` (default) vs. `rerankedSimilarityResponse` (reranked, via `NativeEmbeddingReranker`).
3. **Baseline evaluation** — asked the same 9 questions (factual, causal, relational, and one "trap" question) against a real uploaded document, in both modes, recording full answers and response times.
4. **External fact-check** — verified one specific claim independently (a character's place of origin) against published sources, which revealed reranked mode had produced a fabricated, confidently-stated answer.
5. **Root-cause the failure** — traced why: reranking narrows to fewer, more tightly-scoped context chunks, which appears to make the model's output more terse and confident, without necessarily being more correct.
6. **Implemented a fix** — added a conditional grounding instruction to the system prompt, active only when reranking is on, asking the model to explicitly flag uncertainty when context doesn't clearly support an answer.
7. **Re-evaluated** — reran the same 9 questions post-fix, and found a genuine mixed result: the target hallucination was fixed, but 3 previously-good answers became needlessly hedgy, and one separate known hallucination was untouched.
8. **Documented everything honestly**, including the fix's limitations, rather than only reporting the win.

## Key finding: speed vs. accuracy vs. confidence is a real, three-way tradeoff

| Mode | Speed | Accuracy | Behavior on uncertain questions |
|---|---|---|---|
| Default (plain vector search) | Slower, inconsistent | Comparable | More verbose, more likely to hedge |
| Reranked (unmodified) | Faster on most queries, but not all | Comparable, with one confirmed hallucination | Terser, more confident — including when wrong |
| Reranked + grounding fix | Same as unmodified reranked | Fixed the target hallucination | Over-hedges, including on previously-correct answers |

This is the core, defensible takeaway: **narrower, "more precise" retrieval does not automatically mean more truthful output**, and a blanket prompt-level fix trades one failure mode (overconfident hallucination) for another (excessive, unhelpful hedging). Neither retrieval mode nor this fix is a clean win — which is a more honest and more useful finding than a simple "before/after, it's better now" story.

## The code change

In `server/utils/chats/stream.js`:

```javascript
const groundingInstruction =
  workspace?.vectorSearchMode === "rerank"
    ? "\n\nIMPORTANT: The context above was selected by a precision-focused retrieval process that returns a small, narrow set of passages. If the provided context does not clearly and directly support an answer, explicitly say you are not certain rather than guessing or stating an answer with unwarranted confidence."
    : "";

const finalSystemPrompt = systemPrompt + groundingInstruction;
```

`finalSystemPrompt` replaces the original `systemPrompt` passed into `LLMConnector.compressMessages(...)`. Minimal and contained: no changes to retrieval logic, the reranker itself, or default-mode behavior.

## What I'd do next (not yet implemented)

The blanket instruction is a blunt instrument. A better follow-up would condition the grounding instruction's presence or strength on an actual **retrieval confidence signal** — for example, the reranker's own relevance/similarity scores (already computed inside `rerankedSimilarityResponse`, in the `rerank_score` field) — rather than firing it uniformly whenever reranking happens to be enabled. This would target the instruction at genuinely weak retrieval instead of adding hedging language to every reranked response regardless of retrieval quality.

## Tech stack

Node.js · Ollama (local LLM + embeddings, `llama3.2:3b` / `nomic-embed-text`) · LanceDB (vector database) · Docker · WSL2

## Full evaluation data

See [`anythingllm-reranking-baseline-evaluation.md`](./anythingllm-reranking-baseline-evaluation.md) for the complete question-by-question before/after data, timing, and analysis.

## What this project demonstrates

- Reading and navigating an unfamiliar, real-world (54K+ star) open-source codebase
- Understanding a full RAG pipeline end to end: chunking, embedding, vector search, reranking, prompt assembly
- Designing and running a real evaluation with a control group, not just eyeballing outputs
- Verifying an LLM's claims against external sources rather than trusting confident-sounding text
- Shipping a small, targeted, low-risk code change
- **Honestly reporting a partial result**, including a regression the fix introduced, instead of overstating success
