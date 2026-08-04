"""REPL worker subprocess: JSONL stdio protocol with parent env."""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any

_SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": __import__,
    "open": open,
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}

RESERVED_TOOL_NAMES = frozenset(
    {
        "llm_query",
        "llm_query_batched",
        "rlm_query",
        "rlm_query_batched",
        "SHOW_VARS",
        "answer",
        "context",
        "history",
    }
)


class _AnswerDict(dict):
    def __init__(self, on_ready=None):
        super().__init__()
        super().__setitem__("content", "")
        super().__setitem__("ready", False)
        self._on_ready = on_ready

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "ready" and value and self._on_ready is not None:
            try:
                self._on_ready(self.get("content", ""))
            except Exception:
                pass


class Worker:
    """Execute generated Python while exposing traced LM helper functions.

    The worker persists user locals across code blocks, restores reserved RLM helpers
    after each execution, forwards single/batched calls to the rollout proxy, and adds
    deterministic parent, call-order, batch-index, and helper-kind metadata.

    Args:
        proxy_url: Base URL of ``SubLLMProxy``.
        rollout_id: Rollout handle registered with the proxy.
        depth: Recursion depth exposed in proxy requests.
        exec_timeout_s: Optional per-block timeout; defaults from the environment.

    Example:
        ``result = Worker("http://127.0.0.1:8000", "run").execute("print(1)")``
    """

    def __init__(
        self,
        proxy_url: str,
        rollout_id: str,
        depth: int = 1,
        exec_timeout_s: float | None = None,
    ):
        self.proxy_url = proxy_url.rstrip("/")
        self.rollout_id = rollout_id
        self.depth = depth
        if exec_timeout_s is None:
            try:
                exec_timeout_s = float(os.environ.get("RLM_TRAIN_EXEC_TIMEOUT_S", "600"))
            except ValueError:
                exec_timeout_s = 600.0
        self.exec_timeout_s = exec_timeout_s
        self._lock = threading.Lock()
        self._last_final_answer: str | None = None
        self._context_count = 0
        self._active_trace_context: dict[str, Any] = {}
        self._call_site_counter = 0
        self.globals: dict[str, Any] = {}
        self.locals: dict[str, Any] = {}
        self._setup_namespace()

    def _setup_namespace(self) -> None:
        self.globals = {"__builtins__": _SAFE_BUILTINS.copy(), "__name__": "__main__"}
        self.locals = {}
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["rlm_query"] = self._rlm_query
        self.globals["rlm_query_batched"] = self._rlm_query_batched
        self.locals["answer"] = _AnswerDict(on_ready=self._capture_answer)

    def _restore_scaffold(self) -> None:
        for name in RESERVED_TOOL_NAMES:
            if name == "llm_query":
                self.globals["llm_query"] = self._llm_query
            elif name == "llm_query_batched":
                self.globals["llm_query_batched"] = self._llm_query_batched
            elif name == "rlm_query":
                self.globals["rlm_query"] = self._rlm_query
            elif name == "rlm_query_batched":
                self.globals["rlm_query_batched"] = self._rlm_query_batched
            elif name == "SHOW_VARS":
                self.globals["SHOW_VARS"] = self._show_vars
            elif name == "answer":
                current = self.locals.get("answer")
                if not isinstance(current, _AnswerDict):
                    replacement = _AnswerDict(on_ready=self._capture_answer)
                    if isinstance(current, dict):
                        for k, v in current.items():
                            dict.__setitem__(replacement, k, v)
                        if current.get("ready") and self._last_final_answer is None:
                            self._last_final_answer = str(current.get("content", ""))
                    self.locals["answer"] = replacement
            elif name == "context" and "context_0" in self.locals:
                self.locals["context"] = self.locals["context_0"]

    def _capture_answer(self, content: Any) -> None:
        self._last_final_answer = str(content)

    def _show_vars(self) -> str:
        available = {
            k: type(v).__name__
            for k, v in self.locals.items()
            if not k.startswith("_") and k != "answer"
        }
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    def _proxy_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.proxy_url}/rollout/{self.rollout_id}/{path.lstrip('/')}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = ""
            too_large = (
                e.code == 413
                or "Entity Too Large" in (detail or "")
                or "Entity Too Large" in (e.reason or "")
            )
            return {"error": f"HTTP {e.code}: {detail or e.reason}", "too_large": too_large}
        except Exception as e:
            return {"error": f"Proxy request failed: {e}"}

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        """Execute a single plain LM helper call while retaining its helper identity."""
        del model
        return self._query_single(prompt, call_kind="llm_query")

    def _rlm_query(self, prompt: str, model: str | None = None) -> str:
        """Execute a depth-limited recursive helper call and label it ``rlm_query``."""
        del model
        return self._query_single(prompt, call_kind="rlm_query")

    def _query_single(self, prompt: str, *, call_kind: str) -> str:
        """Send one proxy request with deterministic trace metadata.

        Args:
            prompt: Child question or instruction generated by the policy.
            call_kind: Original helper name used by generated code.

        Returns:
            Child response text, or an explanatory error string for proxy failures.
        """
        result = self._proxy_post(
            "llm_query",
            {
                "prompt": prompt,
                "depth": self.depth,
                "trace_context": self._next_trace_context(call_kind, advance=True),
            },
        )
        if "error" in result and result["error"]:
            if result.get("too_large"):
                return (
                    "Error: sub-LLM prompt exceeded the endpoint's request-size limit "
                    f"(prompt was {len(prompt):,} chars). Shorten or chunk the prompt. "
                    f"Underlying error: {result['error']}"
                )
            return f"Error: {result['error']}"
        return result.get("response", "")

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Execute a batched plain-LM helper call with one shared call-site order."""
        del model
        return self._query_batched(prompts, call_kind="llm_query_batched")

    def _rlm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Execute a batched recursive helper and preserve its helper identity."""
        del model
        return self._query_batched(prompts, call_kind="rlm_query_batched")

    def _query_batched(self, prompts: list[str], *, call_kind: str) -> list[str]:
        """Send aligned prompts and per-item trace contexts to the batch endpoint.

        Args:
            prompts: Child questions generated at one batched call site.
            call_kind: Original batched helper name used by generated code.

        Returns:
            One response or explanatory error string per prompt. Empty input returns an
            empty list without consuming a call-site index.
        """
        if not prompts:
            return []
        trace_contexts = [
            self._next_trace_context(call_kind, advance=False, batch_index=batch_index)
            for batch_index, _ in enumerate(prompts)
        ]
        self._call_site_counter += 1
        result = self._proxy_post(
            "llm_query_batched",
            {
                "prompts": list(prompts),
                "depth": self.depth,
                "trace_contexts": trace_contexts,
            },
        )
        if "error" in result and result["error"]:
            if result.get("too_large"):
                total = sum(len(p) for p in prompts)
                longest = max(len(p) for p in prompts)
                msg = (
                    "Error: sub-LLM batched request exceeded the endpoint's request-size limit "
                    f"({len(prompts)} prompts, total {total:,} chars, longest {longest:,} chars). "
                    f"Underlying error: {result['error']}"
                )
                return [msg] * len(prompts)
            return [f"Error: {result['error']}"] * len(prompts)
        responses = result.get("responses")
        if not isinstance(responses, list) or len(responses) != len(prompts):
            return ["Error: malformed batched response"] * len(prompts)
        return [r if isinstance(r, str) else f"Error: {r}" for r in responses]

    def load_context(self, payload: Any, index: int | None = None) -> int:
        if index is None:
            index = self._context_count
        var = f"context_{index}"
        self.locals[var] = payload
        if index == 0:
            self.locals["context"] = payload
        self._context_count = max(self._context_count, index + 1)
        return index

    @contextmanager
    def _capture_output(self):
        with self._lock:
            old_out, old_err = sys.stdout, sys.stderr
            out_buf, err_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = out_buf, err_buf
                yield out_buf, err_buf
            finally:
                sys.stdout, sys.stderr = old_out, old_err

    def _exec_with_timeout(self, code: str, ns: dict[str, Any]) -> None:
        timeout_s = self.exec_timeout_s
        if timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
            exec(code, ns, ns)  # noqa: S102
            return

        def _on_alarm(signum, frame):  # noqa: ARG001
            raise TimeoutError(f"```repl``` block exceeded {timeout_s:g}s execution timeout")

        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            exec(code, ns, ns)  # noqa: S102
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def _next_trace_context(
        self, call_kind: str, *, advance: bool, batch_index: int | None = None
    ) -> dict[str, Any]:
        """Build trace metadata for the next dynamic call site.

        Args:
            call_kind: Original helper invoked by generated code.
            advance: Whether this item consumes the next call-site index. Single calls
                advance immediately; a batch advances once after all items are labeled.
            batch_index: Optional item index within a batched call.

        Returns:
            A copied context containing parent/block metadata, global call order, helper
            kind, and optional batch index.
        """
        trace_context = dict(self._active_trace_context)
        offset = int(trace_context.pop("call_order_offset", 0))
        trace_context["call_order"] = offset + self._call_site_counter
        trace_context["call_kind"] = call_kind
        if batch_index is not None:
            trace_context["batch_index"] = batch_index
        if advance:
            self._call_site_counter += 1
        return trace_context

    def execute(self, code: str, trace_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute one code block and report outputs plus dynamic trace-call count.

        Args:
            code: Generated Python source to run in the persistent namespace.
            trace_context: Optional parent node, code-block index, and call-order offset.

        Returns:
            JSON-serializable dictionary containing stdout, stderr, final answer,
            execution time, simple local keys, and the number of executed call sites.

        Notes:
            Execution exceptions are captured into ``stderr`` rather than raised across
            the JSONL boundary. Reserved helper names are restored after successful code.
        """
        start = time.perf_counter()
        self._active_trace_context = dict(trace_context or {})
        self._call_site_counter = 0
        with self._capture_output() as (out_buf, err_buf):
            try:
                combined = {**self.globals, **self.locals}
                self._exec_with_timeout(code, combined)
                for k, v in combined.items():
                    if k not in self.globals and not k.startswith("_"):
                        self.locals[k] = v
                self._restore_scaffold()
                stdout = out_buf.getvalue()
                stderr = err_buf.getvalue()
            except BaseException as e:  # noqa: BLE001
                stdout = out_buf.getvalue()
                stderr = err_buf.getvalue() + f"\n{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                if tb and tb.strip() and tb not in stderr:
                    stderr = stderr + "\n" + tb
        final_answer = self._last_final_answer
        self._last_final_answer = None
        trace_call_count = self._call_site_counter
        self._active_trace_context = {}
        simple_keys = [
            k
            for k, v in self.locals.items()
            if not k.startswith("_")
            and k not in ("__builtins__", "__name__", "__doc__")
            and isinstance(v, (str, int, float, bool, list, dict, tuple))
        ]
        return {
            "stdout": stdout,
            "stderr": stderr,
            "final_answer": final_answer,
            "execution_time": time.perf_counter() - start,
            "locals_keys": simple_keys,
            "trace_call_count": trace_call_count,
        }


def _send(obj: dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-url", default=os.environ.get("RLM_TRAIN_PROXY_URL", ""))
    parser.add_argument("--rollout-id", default=os.environ.get("RLM_TRAIN_ROLLOUT_ID", ""))
    parser.add_argument("--depth", type=int, default=int(os.environ.get("RLM_TRAIN_DEPTH", "1")))
    args = parser.parse_args()

    if not args.proxy_url or not args.rollout_id:
        _send({"id": "_init", "ok": False, "error": "missing proxy-url or rollout-id"})
        return

    worker = Worker(proxy_url=args.proxy_url, rollout_id=args.rollout_id, depth=args.depth)
    _send({"id": "_init", "ok": True})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            _send({"id": "?", "ok": False, "error": f"bad json: {e}"})
            continue

        rid = req.get("id", "?")
        kind = req.get("type")

        if kind == "exec":
            try:
                result = worker.execute(req.get("code", ""), req.get("trace_context"))
                _send({"id": rid, "ok": True, **result})
            except BaseException as e:  # noqa: BLE001
                _send(
                    {"id": rid, "ok": False, "error": f"exec failed: {e}\n{traceback.format_exc()}"}
                )
        elif kind == "load_context":
            try:
                idx = worker.load_context(req.get("payload"), req.get("index"))
                _send({"id": rid, "ok": True, "index": idx})
            except BaseException as e:  # noqa: BLE001
                _send({"id": rid, "ok": False, "error": f"load_context failed: {e}"})
        elif kind == "bootstrap":
            code = req.get("code") or ""
            try:
                if code:
                    exec(compile(code, "<bootstrap>", "exec"), worker.globals)
                _send({"id": rid, "ok": True})
            except BaseException as e:  # noqa: BLE001
                _send(
                    {
                        "id": rid,
                        "ok": False,
                        "error": f"bootstrap failed: {e}\n{traceback.format_exc()}",
                    }
                )
        elif kind == "shutdown":
            _send({"id": rid, "ok": True})
            return
        else:
            _send({"id": rid, "ok": False, "error": f"unknown type: {kind!r}"})


if __name__ == "__main__":
    main()
