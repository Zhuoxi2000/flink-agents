################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
"""Tests for embedding model token usage metrics (#858)."""

from typing import Any, Dict, Sequence, Tuple
from unittest.mock import MagicMock

from flink_agents.api.embedding_models.embedding_model import (
    BaseEmbeddingModelConnection,
    BaseEmbeddingModelSetup,
)
from flink_agents.api.metric_group import Counter, MetricGroup


class _MockCounter(Counter):
    def __init__(self) -> None:
        self._count = 0

    def inc(self, n: int = 1) -> None:
        self._count += n

    def dec(self, n: int = 1) -> None:
        self._count -= n

    def get_count(self) -> int:
        return self._count


class _MockMetricGroup(MetricGroup):
    def __init__(self) -> None:
        self._sub_groups: dict[str, _MockMetricGroup] = {}
        self._counters: dict[str, _MockCounter] = {}

    def get_sub_group(self, name: str, value: str | None = None) -> "_MockMetricGroup":
        key = f"{name}={value}" if value is not None else name
        if key not in self._sub_groups:
            self._sub_groups[key] = _MockMetricGroup()
        return self._sub_groups[key]

    def get_counter(self, name: str) -> _MockCounter:
        if name not in self._counters:
            self._counters[name] = _MockCounter()
        return self._counters[name]

    def get_meter(self, name: str) -> Any:
        return MagicMock()

    def get_gauge(self, name: str) -> Any:
        return MagicMock()

    def get_histogram(self, name: str, window_size: int = 100) -> Any:
        return MagicMock()


class _UsageReportingConnection(BaseEmbeddingModelConnection):
    """Connection that reports provider usage, like OpenAI/DashScope."""

    def embed(
        self, text: str | Sequence[str], **kwargs: Any
    ) -> list[float] | list[list[float]]:
        return self.embed_with_usage(text, **kwargs)[0]

    def embed_with_usage(
        self, text: str | Sequence[str], **kwargs: Any
    ) -> Tuple[list[float] | list[list[float]], Dict[str, Any] | None]:
        usage = {
            "model_name": "text-embedding-3-small",
            "promptTokens": 7,
            "totalTokens": 7,
        }
        return [0.1, 0.2], usage


class _NoUsageConnection(BaseEmbeddingModelConnection):
    """Legacy-style connection that only implements embed()."""

    def embed(
        self, text: str | Sequence[str], **kwargs: Any
    ) -> list[float] | list[list[float]]:
        return [0.3, 0.4]


class _Setup(BaseEmbeddingModelSetup):
    @property
    def model_kwargs(self) -> Dict[str, Any]:
        return {"model": self.model}


def _make_setup(connection: BaseEmbeddingModelConnection) -> _Setup:
    setup = _Setup(connection="conn", model="text-embedding-3-small")
    # Bypass open()/resource_context resolution for unit testing.
    setup.connection = connection
    return setup


class TestEmbeddingTokenMetrics:
    def test_usage_recorded_on_bound_metric_group(self) -> None:
        setup = _make_setup(_UsageReportingConnection())
        group = _MockMetricGroup()
        setup.set_metric_group(group)

        result = setup.embed("hello")

        assert result == [0.1, 0.2]
        model_group = group.get_sub_group("model", "text-embedding-3-small")
        assert model_group.get_counter("promptTokens").get_count() == 7
        assert model_group.get_counter("totalTokens").get_count() == 7

    def test_usage_accumulates_across_calls(self) -> None:
        setup = _make_setup(_UsageReportingConnection())
        group = _MockMetricGroup()
        setup.set_metric_group(group)

        setup.embed("hello")
        setup.embed("world")

        model_group = group.get_sub_group("model", "text-embedding-3-small")
        assert model_group.get_counter("promptTokens").get_count() == 14
        assert model_group.get_counter("totalTokens").get_count() == 14

    def test_no_metric_group_is_noop(self) -> None:
        setup = _make_setup(_UsageReportingConnection())

        # No metric group bound: must not raise.
        assert setup.embed("hello") == [0.1, 0.2]

    def test_connection_without_usage_records_nothing(self) -> None:
        setup = _make_setup(_NoUsageConnection())
        group = _MockMetricGroup()
        setup.set_metric_group(group)

        # Default embed_with_usage() keeps legacy connections working.
        assert setup.embed("hello") == [0.3, 0.4]
        assert group._sub_groups == {}

    def test_no_completion_tokens_counter(self) -> None:
        """Embeddings are input-only: no completionTokens counter (#858)."""
        setup = _make_setup(_UsageReportingConnection())
        group = _MockMetricGroup()
        setup.set_metric_group(group)

        setup.embed("hello")

        model_group = group.get_sub_group("model", "text-embedding-3-small")
        assert "completionTokens" not in model_group._counters

    def test_explicit_metric_group_wins_over_rebound_group(self) -> None:
        """Same capture semantics as chat token metrics (#859)."""
        setup = _make_setup(_UsageReportingConnection())
        group_a = _MockMetricGroup()
        group_b = _MockMetricGroup()

        setup.set_metric_group(group_a)
        captured = setup.metric_group
        setup.set_metric_group(group_b)

        setup._record_token_metrics(
            "text-embedding-3-small", 7, 7, metric_group=captured
        )

        model_group = group_a.get_sub_group("model", "text-embedding-3-small")
        assert model_group.get_counter("promptTokens").get_count() == 7
        assert group_b._sub_groups == {}
