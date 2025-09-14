Here’s a concise, high-level take on each point:

1. Differences between “completion” and “chat” models

* Completion: single text-in → text-out; no roles; you craft the whole prompt. Good for templated generations and legacy APIs.
* Chat: multi-turn with roles (system/user/assistant), built-in conversation handling and safety. Better instruction following, tool use, and memory.

2. Reasoning model vs generalist model

* Generalist: fast, broad capability; answers directly from the prompt+context.
* Reasoning: trained/optimized to think step-by-step (internal reasoning, tool use, deliberate decoding). Pros: harder tasks, better reliability; Cons: slower, costlier.

3. Forcing “yes/no” and parsing fixed formats

* Prompting: “Answer strictly ‘yes’ or ‘no’.” Set temperature low.
* Constrained output: JSON schema/function-calling, Pydantic/Output Parsers, or grammar-constrained decoding (EBNF) so only valid shapes are allowed.
* Logit bias (if API supports) to allow only “yes”/“no”.
* Post-parse: validate with a schema; on failure, reprompt with the parse error.

4. RAG vs fine-tuning (when/why; pros/cons)

* RAG (retrieve-then-generate):

  * Use when knowledge lives in documents & must stay fresh/traceable.
  * Pros: up-to-date, cheaper, grounded answers with citations.
  * Cons: needs good retrieval; can fail if docs or chunks are poor.
* Fine-tuning:

  * Use to adapt style, format, domain behaviors; when patterns recur and docs aren’t needed at runtime.
  * Pros: fast inference, smaller prompts, domain tone.
  * Cons: static (needs retraining to update), cost to train, risk of memorization.

5. What is an agent?

* An LLM loop that can plan → call tools (search, code, DB) → observe results → decide next step until a goal is reached.
* Think “LLM + tools + memory + policy” rather than a single response.

6. Evaluating Q\&A bots, RAG, and GenAI apps (tools & metrics)

* Q\&A bot (end-to-end): exact match / F1 on golden sets, human rated helpfulness, latency, cost, safety (toxicity/jailbreak).
* RAG (component + end-to-end):

  * Retrieval: Recall\@k, Precision\@k, MRR / nDCG, context hit rate (answer found in context).
  * Generation: faithfulness/groundedness (does it cite provided context?), answer correctness (LLM-as-judge + spot-checks), citation accuracy.
* GenAI app (overall): task success rate, consistency/regression tests, UX metrics (time-to-answer), observability (traces, prompts, tool calls).
* Useful tooling: Ragas, DeepEval, TruLens, LangChain Benchmarks, Evals with golden datasets, A/B tests, prompt unit tests, telemetry dashboards.
