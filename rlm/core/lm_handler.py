"""
LMHandler - Routes LLM requests from the RLM process and environment subprocesses.

Uses a multi-threaded socket server. Protocol: 4-byte length prefix + JSON payload.
"""

import asyncio
import time
from collections.abc import Callable
from socketserver import StreamRequestHandler, ThreadingTCPServer
from threading import Thread
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.comms_utils import LMRequest, LMResponse, socket_recv, socket_send
from rlm.core.types import RLMChatCompletion, UsageSummary


class LMRequestHandler(StreamRequestHandler):
    """Socket handler for LLM completion requests."""

    def handle(self):
        try:
            request_data = socket_recv(self.connection)
            if not isinstance(request_data, dict):
                response = LMResponse.error_response("Request must be a JSON object")
                self._safe_send(response)
                return

            request = LMRequest.from_dict(request_data)
            handler: LMHandler = self.server.lm_handler  # type: ignore

            if request.is_batched:
                # Batched request: process multiple prompts concurrently
                response = self._handle_batched(request, handler)
            elif request.prompt:
                # Single request: process one prompt
                response = self._handle_single(request, handler)
            else:
                response = LMResponse.error_response("Missing 'prompt' or 'prompts' in request.")

            self._safe_send(response)

        except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError):
            # Client disconnected - this is expected during parallel execution
            # when workers complete and close their sockets. Silently ignore.
            pass

        except Exception as e:
            # Try to send error response, but don't fail if socket is broken
            response = LMResponse.error_response(str(e))
            self._safe_send(response)

    def _safe_send(self, response: LMResponse) -> bool:
        """Send response, returning False if the socket is broken."""
        try:
            socket_send(self.connection, response.to_dict())
            return True
        except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError):
            # Client disconnected - silently ignore
            return False

    def _handle_single(self, request: LMRequest, handler: "LMHandler") -> LMResponse:
        """Handle a single prompt request."""
        client = handler.get_client(request.model, request.depth)
        event_context = handler.plain_subcall_started(
            request.prompt, request.model, request.depth, None
        )

        start_time = time.perf_counter()
        try:
            content = client.completion(request.prompt)
        except Exception as exc:
            handler.plain_subcall_completed(
                event_context,
                RLMChatCompletion(
                    root_model=request.model or client.model_name,
                    prompt=request.prompt,
                    response="",
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=time.perf_counter() - start_time,
                    error=str(exc),
                ),
            )
            raise
        end_time = time.perf_counter()

        model_usage = client.get_last_usage()
        root_model = request.model or client.model_name
        usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})
        generation = client.get_last_generation()
        if not isinstance(generation, dict):
            generation = {}
        completion = RLMChatCompletion(
            root_model=root_model,
            prompt=request.prompt,
            response=content,
            usage_summary=usage_summary,
            execution_time=end_time - start_time,
            prompt_token_ids=tuple(generation.get("prompt_token_ids") or ()),
            token_ids=tuple(generation.get("token_ids") or ()),
            token_offsets=tuple(tuple(offset) for offset in generation.get("token_offsets") or ()),
            prompt_token_count=generation.get("prompt_token_count"),
            policy_owner=generation.get("policy_owner"),
        )
        handler.plain_subcall_completed(event_context, completion)
        return LMResponse.success_response(chat_completion=completion)

    def _handle_batched(self, request: LMRequest, handler: "LMHandler") -> LMResponse:
        """Handle a batched prompts request using async for concurrency."""
        client = handler.get_client(request.model, request.depth)

        start_time = time.perf_counter()
        event_contexts = [
            handler.plain_subcall_started(prompt, request.model, request.depth, index)
            for index, prompt in enumerate(request.prompts)
        ]

        sem = asyncio.Semaphore(handler.batch_max_concurrent)

        async def run_one(prompt: str):
            async with sem:
                return await client.acompletion_with_generation(prompt)

        async def run_all():
            tasks = [run_one(prompt) for prompt in request.prompts]
            # return_exceptions=True so one failed call doesn't abort the whole
            # batch; failures are surfaced per-prompt as error completions below.
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(run_all())
        end_time = time.perf_counter()

        total_time = end_time - start_time
        model_usage = client.get_last_usage()
        root_model = request.model or client.model_name
        usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})

        chat_completions = []
        for prompt, result, event_context in zip(
            request.prompts, results, event_contexts, strict=True
        ):
            if isinstance(result, BaseException):
                # Per-prompt failure: this slot returns an error; other prompts
                # still succeed. The error message is carried back to the caller.
                completion = RLMChatCompletion(
                    root_model=root_model,
                    prompt=prompt,
                    response="",
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                    error=f"llm() call failed - {result}",
                )
                chat_completions.append(completion)
                handler.plain_subcall_completed(event_context, completion)
            else:
                content, generation = result
                generation = generation or {}
                completion = RLMChatCompletion(
                    root_model=root_model,
                    prompt=prompt,
                    response=content,
                    usage_summary=usage_summary,
                    execution_time=total_time / len(request.prompts),  # approximate per-prompt time
                    prompt_token_ids=tuple(generation.get("prompt_token_ids") or ()),
                    token_ids=tuple(generation.get("token_ids") or ()),
                    token_offsets=tuple(
                        tuple(offset) for offset in generation.get("token_offsets") or ()
                    ),
                    prompt_token_count=generation.get("prompt_token_count"),
                    policy_owner=generation.get("policy_owner"),
                )
                chat_completions.append(completion)
                handler.plain_subcall_completed(event_context, completion)

        return LMResponse.batched_success_response(chat_completions=chat_completions)


class ThreadingLMServer(ThreadingTCPServer):
    """Multi-threaded TCP server for LM requests."""

    daemon_threads = True
    allow_reuse_address = True


class LMHandler:
    """
    Handles all LM calls from the RLM main process and environment subprocesses.

    Uses a multi-threaded socket server for concurrent requests.
    Protocol: 4-byte big-endian length prefix + JSON payload.
    """

    def __init__(
        self,
        client: BaseLM,
        host: str = "127.0.0.1",
        port: int = 0,  # auto-assign available port
        other_backend_client: BaseLM | None = None,
        batch_max_concurrent: int = 16,
        on_plain_subcall_start: Callable[[Any, str | None, int, int | None], Any] | None = None,
        on_plain_subcall_complete: Callable[[Any, RLMChatCompletion], None] | None = None,
    ):
        self.default_client = client
        self.other_backend_client = other_backend_client
        self.clients: dict[str, BaseLM] = {}
        self.host = host
        self._server: ThreadingLMServer | None = None
        self._thread: Thread | None = None
        self._port = port
        self.batch_max_concurrent = batch_max_concurrent
        self.on_plain_subcall_start = on_plain_subcall_start
        self.on_plain_subcall_complete = on_plain_subcall_complete

        self.register_client(client.model_name, client)

    def register_client(self, model_name: str, client: BaseLM) -> None:
        """Register a client for a specific model name."""
        self.clients[model_name] = client

    def plain_subcall_started(
        self, prompt: Any, model: str | None, depth: int, batch_index: int | None
    ) -> Any:
        if self.on_plain_subcall_start is None:
            return None
        return self.on_plain_subcall_start(prompt, model, depth, batch_index)

    def plain_subcall_completed(self, event_context: Any, completion: RLMChatCompletion) -> None:
        if self.on_plain_subcall_complete is not None:
            self.on_plain_subcall_complete(event_context, completion)

    def get_client(self, model: str | None = None, depth: int = 0) -> BaseLM:
        """Get client by model name or depth, or return default.

        Routing logic:
        - depth=0: use default_client (main backend)
        - depth=1: use other_backend_client if it exists, otherwise default_client
        - If model is specified and exists in clients, use that (overrides depth routing)
        """
        if model and model in self.clients:
            return self.clients[model]

        # Route based on depth
        if depth == 1 and self.other_backend_client is not None:
            return self.other_backend_client

        return self.default_client

    @property
    def port(self) -> int:
        """Get the actual port (useful when auto-assigned)."""
        if self._server:
            return self._server.server_address[1]
        return self._port

    @property
    def address(self) -> tuple[str, int]:
        """Get (host, port) tuple for connecting."""
        return (self.host, self.port)

    def start(self) -> tuple[str, int]:
        """Start the socket server in a background thread. Returns (host, port)."""
        if self._server is not None:
            return self.address

        self._server = ThreadingLMServer((self.host, self._port), LMRequestHandler)
        self._server.lm_handler = self  # type: ignore

        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        return self.address

    def stop(self):
        """Stop the socket server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None

    def completion(self, prompt: str, model: str | None = None) -> str:
        """Direct completion call (for main process use)."""
        return self.get_client(model).completion(prompt)

    def get_last_generation(self, model: str | None = None) -> dict | None:
        """Return exact sampled-ID metadata exposed by the selected client."""
        generation = self.get_client(model).get_last_generation()
        return generation if isinstance(generation, dict) else None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def get_usage_summary(self) -> UsageSummary:
        """Get the usage summary for all clients, merged into a single dict."""
        merged = {}
        # Include default client
        default_summary = self.default_client.get_usage_summary()
        merged.update(default_summary.model_usage_summaries)
        # Include other backend client if it exists
        if self.other_backend_client is not None:
            other_summary = self.other_backend_client.get_usage_summary()
            merged.update(other_summary.model_usage_summaries)
        # Include all registered clients
        for client in self.clients.values():
            client_summary = client.get_usage_summary()
            merged.update(client_summary.model_usage_summaries)
        return UsageSummary(model_usage_summaries=merged)
