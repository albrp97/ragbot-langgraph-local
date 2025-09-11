from typing import Dict, Any
from app.llm.chat import generate
from app.tools.python_tool import _extract_code_block, safe_exec

CODE_PROMPT = """You are a Python coding assistant.
Write a minimal Python snippet to answer the user's question.
Rules:
- No imports.
- You may use built-ins, math, statistics, random.
- Do not read/write files or network.
- Print the final answer OR assign it to a variable named `result`.
- Output ONLY a Python code block.

# Question
{question}
"""

def run_python_for_question(question: str, timeout_s: float = 3.0) -> Dict[str, Any]:
    # Generate code from the question
    # timeout_s is for code execution, not generation
    # Overrides: temperature=0.0, top_p=1.0 for deterministic output, and thinking=True for progress indicator
    code_text = generate(CODE_PROMPT.format(question=question), thinking=True, do_sample=False)
    code = _extract_code_block(code_text)
    exec_res = safe_exec(code, timeout_s=timeout_s)
    return {"code": code, **exec_res}
