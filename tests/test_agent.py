"""Tests for how the agent calls the model and handles its replies.

WHAT: The request shape (model, max_tokens, effort, JSON schema), the stop-reason
      check, the fallbacks of `select` and `make_plan`, and the failure log line.
HOW:  A fake client stands in for `anthropic.Anthropic`: it records the request and
      returns a canned reply object with `stop_reason`, `content` and `usage`. No
      network and no API key are needed.
WHY:  A live run fell back to ranking because the gate's reply was not valid JSON and
      the log did not say why. These tests pin the fix: structured output on both JSON
      calls, room in max_tokens for thinking plus the reply, a refusal or cut-off reply
      treated as a failure, and a log line that shows the stop reason and the reply.
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from openbell.agent import (PLAN_SCHEMA, SELECT_SCHEMA, Agent, build_request, reply_text)
from openbell.config import Config
from openbell.models import DailyContext, Session
from openbell.scanner import Candidate

UTC = timezone.utc


def reply(text, stop="end_turn"):
    """A fake Messages API response: one thinking block, then the text."""
    return SimpleNamespace(stop_reason=stop,
                           content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))


class FakeClient:
    """Records each request and returns the queued responses in order."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        return self.responses.pop(0)


def agent_with(*responses):
    a = Agent(Config())                     # no key in the test env -> client is None
    a.client = FakeClient(*responses)
    return a


def cands(*syms):
    return [Candidate(s, "SWING", 1.0, rank=i + 1) for i, s in enumerate(syms)]


def test_request_carries_effort_schema_and_room_for_thinking():
    cfg = Config()
    req = build_request(cfg, "sys", "user", SELECT_SCHEMA)
    assert req["model"] == "claude-haiku-5-5" and req["max_tokens"] >= 16000
    assert req["output_config"] == {"effort": "medium",
                                    "format": {"type": "json_schema", "schema": SELECT_SCHEMA}}
    assert "format" not in build_request(cfg, "sys", "user")["output_config"]   # review: free text


def test_schemas_follow_structured_output_rules():
    """Every object closes additionalProperties and requires all its fields."""
    def check(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                check(child)
        if node.get("type") == "array":
            check(node["items"])
    check(SELECT_SCHEMA)
    check(PLAN_SCHEMA)


def test_reply_text_rejects_unfinished_replies():
    assert reply_text(reply('{"a": 1}')) == '{"a": 1}'           # thinking block skipped
    for stop in ("max_tokens", "refusal"):
        with pytest.raises(ValueError, match=stop):
            reply_text(reply('{"decisions": [', stop))


def test_select_uses_schema_and_parses_list_reply():
    a = agent_with(reply(json.dumps({"decisions": [
        {"ticker": "AAA", "approve": False, "reason": "share offering"},
        {"ticker": "BBB", "approve": True, "reason": "clean"}]})))
    assert a.select(cands("AAA", "BBB")) == {"AAA": (False, "share offering"), "BBB": (True, "clean")}
    sent = a.client.requests[0]
    assert sent["output_config"]["format"]["schema"] is SELECT_SCHEMA
    assert sent["output_config"]["effort"] == "medium"


def test_select_cut_off_reply_falls_back_and_logs_why(caplog):
    a = agent_with(reply('{"decisions": [{"ticker": "AAA", "approve": true', "max_tokens"))
    with caplog.at_level(logging.WARNING, logger="openbell.agent"):
        assert a.select(cands("AAA")) is None                   # runner then ranks by score
    assert "stop_reason=max_tokens" in caplog.text


def test_select_bad_json_logs_the_reply(caplog):
    a = agent_with(reply('{"decisions": [{"ticker": "AAA", "reason": "CEO "strong" quarter"}]}'))
    with caplog.at_level(logging.WARNING, logger="openbell.agent"):
        assert a.select(cands("AAA")) is None
    assert "JSONDecodeError" in caplog.text and "CEO" in caplog.text   # reply shown for diagnosis


def test_make_plan_parses_schema_reply_and_falls_back_on_refusal():
    o = datetime(2026, 11, 10, 14, 30, tzinfo=UTC)
    sess = Session(o.date(), o, o + timedelta(hours=6, minutes=30))
    ctx = {"AAA": DailyContext("AAA", 100, 101, 99, 2.0, 99, 95, 1, 3, 2)}
    snap = {"equity": 1, "cash": 1, "total_pnl_pct": 0, "exposure_pct": 0, "positions": []}
    good = reply(json.dumps({"regime": "risk_off", "risk_multiplier": 0.4, "summary": "weak tape",
                             "symbols": [{"ticker": "AAA", "allow_intraday": False,
                                          "allow_swing": True, "note": "gap risk"}]}))
    a = agent_with(good, reply("", "refusal"))
    p = a.make_plan(sess, ctx, snap, {}, [])
    assert (p.source, p.regime, p.risk_multiplier) == ("agent", "risk_off", 0.4)
    assert p.symbols["AAA"].allow_intraday is False
    assert a.client.requests[0]["output_config"]["format"]["schema"] is PLAN_SCHEMA
    fallback = a.make_plan(sess, ctx, snap, {}, [])
    assert fallback.source == "default" and "ValueError" in fallback.summary
