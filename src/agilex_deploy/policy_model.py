"""Bounded OpenRouter retries; model requests never execute robot actions."""

import json
import random
import time
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def require_active_session(observed):
    """Cancel inference/actions with the underlying session fault preserved."""
    if not observed["session_enabled"] or observed.get("session_fault"):
        reason = observed.get("session_fault") or "session is stopped or not enabled"
        raise RuntimeError(f"Session stopped or faulted: {reason}")


def add_model_arguments(parser):
    parser.add_argument("--model", default="google/gemini-3.8-flash")
    parser.add_argument("--reasoning-effort", choices=("default", "low", "medium", "high"),
                        default="default", help="Model reasoning effort; default leaves the provider setting unchanged")
    parser.add_argument("--provider-sort", choices=("default", "latency", "throughput", "price"),
                        default="latency", help="OpenRouter provider priority; provider fallbacks remain enabled")


def model_options(args):
    provider = {"allow_fallbacks": True}
    sorting = getattr(args, "provider_sort", "latency")
    if sorting != "default":
        provider["sort"] = sorting
    options = {"provider": provider}
    effort = getattr(args, "reasoning_effort", "default")
    if effort != "default":
        options["reasoning"] = {"effort": effort}
    return options


def response_metadata(result):
    """Record termination and token usage without logging private reasoning."""
    choice = (result.get("choices") or [{}])[0]
    metadata = dict(finish_reason=choice.get("finish_reason"),
                native_finish_reason=choice.get("native_finish_reason"),
                usage=result.get("usage"))
    for field in ("model", "provider"):
        if result.get(field):
            metadata[field] = result[field]
    return metadata


class ModelResponseGuard:
    """Reject empty or truncated turns before any tools or code are executed.

    Retries use the runner's existing turn budget. An unusable response never
    means task completion, and partial tool calls must never command hardware.
    """
    def __init__(self):
        self.max_tokens = 4096
        self.invalid_responses = 0

    def message(self, result, trace):
        choices = result.get("choices") or []
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise RuntimeError("Model returned no assistant message")
        choice = choices[0]
        message = choice["message"]
        reason = choice.get("finish_reason")
        if reason in ("error", "content_filter"):
            raise RuntimeError(f"Model response stopped with finish_reason={reason}")
        empty = not (message.get("content") or "").strip() and not message.get("tool_calls")
        if reason == "length" or empty:
            self.invalid_responses += 1
            issue = "truncated" if reason == "length" else "empty"
            if self.invalid_responses >= 3:
                raise RuntimeError(f"Model returned {self.invalid_responses} consecutive unusable responses ({issue}); task is not complete")
            self.max_tokens = min(self.max_tokens * 2, 16384)
            trace.emit("model_retry", reason=issue, attempt=self.invalid_responses,
                       next_max_tokens=self.max_tokens)
            return None
        self.invalid_responses = 0
        return message


class ModelRequestError(RuntimeError):
    def __init__(self, status, message, provider=None):
        self.status = status
        self.provider = provider
        super().__init__(f"OpenRouter {status or 'network error'}: {message}")


def _api_error(body, status, request):
    error = body.get("error", {}) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        error = {"message": str(error)}
    metadata = error.get("metadata") or {}
    message = str(error.get("message") or "Request failed")
    # Only log selected error fields, never the request body or authorization.
    key = (request.get_header("Authorization") or "").removeprefix("Bearer ")
    if key:
        message = message.replace(key, "[REDACTED]")
    code = error.get("code", status)
    try:
        code = int(code)
    except (ValueError, TypeError):
        code = status
    return ModelRequestError(code, message[:1000], metadata.get("provider_name"))


def _retry_after(headers):
    value = headers.get("Retry-After") if headers else None
    if value is None:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return 0.0


def _wait_for_retry(delay, check_active):
    deadline = time.monotonic() + delay
    while True:
        check_active()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(.25, remaining))


def request_model(request, trace, check_active):
    """Retry inference only. No tool or code can run before a complete response.

    Preserve the exact conversation, tool IDs, and model across attempts. The
    caller executes actions once, after this function returns and checks STOP.
    """
    for attempt in range(3):
        check_active()
        retry_after = 0.0
        try:
            with urlopen(request, timeout=45) as response:
                result = json.load(response)
            # OpenRouter can return provider failures inside an HTTP 200 body.
            if isinstance(result, dict) and result.get("error"):
                raise _api_error(result, None, request)
        except HTTPError as error:
            retry_after = _retry_after(error.headers)
            try:
                body = json.loads(error.read(65536))
            except (ValueError, OSError):
                body = {"error": {"message": error.reason}}
            finally:
                error.close()
            failure = _api_error(body, error.code, request)
        except ModelRequestError as error:
            failure = error
        except (URLError, TimeoutError) as error:
            failure = ModelRequestError(None, str(error))
        else:
            check_active()
            return result
        trace.emit("model_request_failed", attempt=attempt + 1, status=failure.status,
                   provider=failure.provider, error=str(failure))
        retryable = failure.status is None or failure.status in (408, 429, 500, 502, 503, 504)
        if not retryable or attempt == 2 or retry_after > 60:
            raise failure
        delay = max(retry_after, 2 ** (attempt + 1) + random.uniform(0, .5))
        trace.emit("model_retry", attempt=attempt + 1, delay_s=delay, error=str(failure))
        _wait_for_retry(delay, check_active)
    raise AssertionError("Unreachable")
