from __future__ import annotations
import ast, re, time, threading, io, contextlib
from typing import Tuple, Dict, Any
import math, statistics, random

# Pattern for forbidden tokens (imports, IO, system operations) for safety
FORBIDDEN_PATTERN = re.compile(
    r"(^|\W)(?:import|exec|eval|compile|__import__|open|os|sys|subprocess|socket|requests|shutil|pathlib|pickle|builtins)\b",
    flags=re.IGNORECASE,
)

# Safe built-ins to allow in the execution environment
SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "range": range, "enumerate": enumerate, "sorted": sorted,
    "map": map, "filter": filter, "list": list, "dict": dict,
    "set": set, "tuple": tuple, "print": print, "round": round,
}

# Safe modules to allow in the execution environment
SAFE_MODULES = {"math": math, "statistics": statistics, "random": random}

def _extract_code_block(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: single backticks or raw
    m2 = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    return (m2.group(1).strip() if m2 else text.strip())

def _validate(code: str) -> Tuple[bool, str]:
    if FORBIDDEN_PATTERN.search(code):
        return False, "Forbidden token found (imports / IO / system operations are blocked)."
    try:
        tree = ast.parse(code, mode="exec")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "Imports are not allowed."
            if isinstance(node, ast.Attribute):
                # block dunder attributes
                if isinstance(node.attr, str) and node.attr.startswith("__"):
                    return False, "Dunder attributes are not allowed."
    except Exception as e:
        return False, f"Invalid Python code: {e}"
    return True, ""

def safe_exec(code: str, timeout_s: float = 3.0) -> Dict[str, Any]:
    ok, msg = _validate(code)
    if not ok:
        return {"ok": False, "error": msg, "stdout": "", "result": None, "code": code}

    env = {"__builtins__": SAFE_BUILTINS}
    env.update(SAFE_MODULES)

    buf = io.StringIO()
    result_holder = {"exc": None}

    def _runner():
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, env, env)
        except Exception as e:
            result_holder["exc"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        return {"ok": False, "error": f"Timeout > {timeout_s}s", "stdout": buf.getvalue(), "result": None, "code": code}

    if result_holder["exc"] is not None:
        return {"ok": False, "error": str(result_holder["exc"]), "stdout": buf.getvalue(), "result": None, "code": code}

    return {"ok": True, "error": "", "stdout": buf.getvalue(), "result": env.get("result", None), "code": code}
