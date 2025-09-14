
### Completion and chat models

- Completion models do not have context of any older messages, they complete text without memory
- Chat models are trained with instructions, dislogs and roles. They work with multi turn context

### Reasoning and generalistic models

- Generalist models are trained for several task, they will solve superficial tasks and are usually used for conversation since the latency per response is lower
- Reasoning models are finetuned with a chain of thought and instructions step by step to solve precise tasks. They are slower as they generate more tokens but are better at solving logic, math and planning prompts

### Forcing yes/no and formatted outputs

- Clear instructions in the prompt (system or user)
- Examples of responses in the prompt
- Regex parsing after generation, json validations, expected tokens
- Second run with another prompt only for formatting

### RAG vs fine tuning

#### RAG

- Up to date information
- Model agnostic
- Cheaper, more flexible, easier to update
- Source citation
- Depends on retrieval performance
- Higher latency

#### Fine tuning

- Learns style and specific tasks and outputs
- Lower latency
- High effort and high cost to update
- Requires quality data
- Risk of memorization

### Agent

- AI system that uses an LLM as core with capabilities to use tool (APIs, databsaes, code, etc), planning and loops decision making until an objetive is completed

### Performace evaluation

#### Q&A bot

- Ground truth testing dataset, exact match and f1
- Compare context for hallucinations
- Human evaluation and automated with a comparison model
- Rated helpfulness
- Latency, cost, jailbreak score

#### RAG

- Retrieval - Context relevance measuring (hit rate) if the answer is found in context
- Generation - Groundedness citing the provided context and answer correctness with a comparison LLM

#### Gen AI

- Metrics like Rouge, Bleu, BertScore, for summary or translation
- Perplexity metric for LLMs
- Fact checking with a 'better' model and cross checking making the model explain the reasoning or sources
- Guardrails and pydantic for formatting

#### Tools

 - Ragas
 - DeepEval
 - TruLens
