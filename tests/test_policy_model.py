"""Malformed model output must not finish a policy or execute partial actions."""

import argparse
import io
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request

from agilex_deploy.policy_model import (ModelRequestError, ModelResponseGuard,
    add_model_arguments, model_options, request_model, response_metadata,
    require_active_session)


class ModelResponseTests(unittest.TestCase):
    def test_session_cancellation_preserves_the_original_fault(self):
        fault = "Camera failure: RGB-D acquisition is stale"
        for enabled in (False, True):
            with self.assertRaisesRegex(RuntimeError, fault):
                require_active_session(dict(session_enabled=enabled, session_fault=fault))
        with self.assertRaisesRegex(RuntimeError, "stopped or not enabled"):
            require_active_session(dict(session_enabled=False, session_fault=None))
        require_active_session(dict(session_enabled=True, session_fault=None))

    def test_repeated_empty_responses_fail_with_bounded_retries(self):
        guard, trace = ModelResponseGuard(), Mock()
        response = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
        self.assertIsNone(guard.message(response, trace))
        self.assertIsNone(guard.message(response, trace))
        with self.assertRaisesRegex(RuntimeError, "task is not complete"):
            guard.message(response, trace)
        self.assertEqual(trace.emit.call_count, 2)

    def test_truncated_tool_call_is_never_returned_for_execution(self):
        guard = ModelResponseGuard()
        partial = {"content": None, "tool_calls": [{"function": {
            "name": "motion_move_tcp", "arguments": '{"arm": "left"'}}]}
        self.assertIsNone(guard.message({"choices": [{"message": partial,
                                                     "finish_reason": "length"}]}, Mock()))

    def test_valid_tools_with_null_content_remain_valid(self):
        message = {"content": None, "tool_calls": [{"function": {
            "name": "robot_observe", "arguments": "{}"}}]}
        result = {"choices": [{"message": message, "finish_reason": "tool_calls"}]}
        self.assertEqual(ModelResponseGuard().message(result, Mock()), message)

    def test_response_metadata_excludes_reasoning_content(self):
        result = {"choices": [{"message": {"reasoning": "private"},
                               "finish_reason": "length"}], "usage": {"completion_tokens": 4096}}
        self.assertEqual(response_metadata(result), dict(finish_reason="length",
            native_finish_reason=None, usage={"completion_tokens": 4096}))


class ModelTransportTests(unittest.TestCase):
    def setUp(self):
        self.request = Request("https://openrouter.ai/api/v1/chat/completions",
            data=b'{"messages": [{"role": "tool", "tool_call_id": "completed-once", "content": "done"}]}',
            headers={"Authorization": "Bearer test-private-key"})
        self.trace = Mock()
        self.success = {"choices": [{"message": {"content": "Done"}, "finish_reason": "stop"}]}

    def response(self, body=None):
        return io.BytesIO(json.dumps(self.success if body is None else body).encode())

    def failure(self, code=504, headers=None):
        return HTTPError(self.request.full_url, code, "Gateway timeout", headers or {},
            self.response({"error": {"code": code, "message": "Provider timeout test-private-key",
                                     "metadata": {"provider_name": "Test provider"}}}))

    def test_retry_preserves_conversation_and_does_not_replay_tools(self):
        check = Mock()
        with patch("agilex_deploy.policy_model.urlopen", side_effect=[self.failure(), self.response()]) as http, \
             patch("agilex_deploy.policy_model._wait_for_retry") as wait:
            result = request_model(self.request, self.trace, check)
        self.assertEqual(result, self.success)
        self.assertEqual(http.call_count, 2)
        self.assertIs(http.call_args_list[0].args[0], self.request)
        self.assertIs(http.call_args_list[1].args[0], self.request)
        self.assertGreaterEqual(check.call_count, 3)
        wait.assert_called_once()
        self.assertNotIn("test-private-key", str(self.trace.mock_calls))
        self.assertIn("Test provider", str(self.trace.mock_calls))

    def test_retry_exhaustion_is_bounded_and_backoff_increases(self):
        with patch("agilex_deploy.policy_model.urlopen", side_effect=[self.failure() for _ in range(3)]) as http, \
             patch("agilex_deploy.policy_model.random.uniform", return_value=0), \
             patch("agilex_deploy.policy_model._wait_for_retry") as wait:
            with self.assertRaisesRegex(ModelRequestError, "OpenRouter 504"):
                request_model(self.request, self.trace, Mock())
        self.assertEqual(http.call_count, 3)
        self.assertEqual([c.args[0] for c in wait.call_args_list], [2, 4])

    def test_auth_failure_is_not_retried(self):
        with patch("agilex_deploy.policy_model.urlopen", side_effect=self.failure(401)) as http:
            with self.assertRaises(ModelRequestError):
                request_model(self.request, self.trace, Mock())
        self.assertEqual(http.call_count, 1)

    def test_http_200_provider_error_is_retried(self):
        failure = self.response({"error": {"code": 504, "message": "Provider timed out"}})
        with patch("agilex_deploy.policy_model.urlopen", side_effect=[failure, self.response()]), \
             patch("agilex_deploy.policy_model._wait_for_retry"):
            self.assertEqual(request_model(self.request, self.trace, Mock()), self.success)

    def test_retry_after_is_respected(self):
        with patch("agilex_deploy.policy_model.urlopen", side_effect=[self.failure(429, {"Retry-After": "12"}), self.response()]), \
             patch("agilex_deploy.policy_model._wait_for_retry") as wait:
            request_model(self.request, self.trace, Mock())
        self.assertEqual(wait.call_args.args[0], 12)

    def test_stop_during_backoff_prevents_second_request(self):
        check = Mock(side_effect=[None, RuntimeError("Operator STOP")])
        with patch("agilex_deploy.policy_model.urlopen", side_effect=self.failure()) as http:
            with self.assertRaisesRegex(RuntimeError, "Operator STOP"):
                request_model(self.request, self.trace, check)
        self.assertEqual(http.call_count, 1)

    def test_stop_after_response_discards_it_before_execution(self):
        check = Mock(side_effect=[None, RuntimeError("Operator STOP")])
        with patch("agilex_deploy.policy_model.urlopen", return_value=self.response()):
            with self.assertRaisesRegex(RuntimeError, "Operator STOP"):
                request_model(self.request, self.trace, check)
        self.trace.emit.assert_not_called()

    def test_low_latency_defaults_can_be_overridden_explicitly(self):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        self.assertEqual(model_options(parser.parse_args([])), {
            "provider": {"sort": "latency", "allow_fallbacks": True}})
        self.assertEqual(model_options(parser.parse_args(["--reasoning-effort", "low"]))["reasoning"],
                         {"effort": "low"})
        self.assertEqual(model_options(parser.parse_args([
            "--reasoning-effort", "default", "--provider-sort", "default"])),
            {"provider": {"allow_fallbacks": True}})
