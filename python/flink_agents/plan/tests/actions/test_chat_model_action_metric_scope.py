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
"""Regression test for #859: action-scoped token metrics must be attributed to
the action that issued the chat request, even when another action re-acquires
the same cached chat model resource before the metrics are recorded.
"""

import asyncio
from typing import Any, Dict, Sequence
from unittest.mock import MagicMock
from uuid import uuid4

from flink_agents.api.chat_message import ChatMessage, MessageRole
from flink_agents.api.chat_models.chat_model import BaseChatModelSetup
from flink_agents.api.core_options import (
    AgentExecutionOptions,
    ErrorHandlingStrategy,
)
from flink_agents.api.resource import ResourceType
from flink_agents.plan.actions.chat_model_action import chat
from flink_agents.plan.tests.actions.test_chat_model_action_retry import (
    _MockMemoryObject,
    _MockMetricGroup,
)


class _SharedChatModelSetup(BaseChatModelSetup):
    """Chat model that reports token usage via extra_args, like real setups."""

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def resource_type(cls) -> ResourceType:
        return ResourceType.CHAT_MODEL

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatMessage:
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="ok",
            extra_args={
                "model_name": "mock-model",
                "promptTokens": 100,
                "completionTokens": 50,
            },
        )


def _create_ctx(
    chat_model: _SharedChatModelSetup, action_group: _MockMetricGroup
) -> MagicMock:
    config = MagicMock()
    option_values = {
        id(AgentExecutionOptions.ERROR_HANDLING_STRATEGY): ErrorHandlingStrategy.FAIL,
        id(AgentExecutionOptions.CHAT_ASYNC): False,
    }
    config.get = MagicMock(
        side_effect=lambda option: option_values.get(
            id(option), option.get_default_value()
        )
    )

    ctx = MagicMock()
    ctx.config = config
    ctx.sensory_memory = _MockMemoryObject()
    ctx.action_metric_group = action_group
    ctx.send_event = MagicMock()

    def _get_resource(name: str, type: ResourceType) -> _SharedChatModelSetup:
        # Mirrors FlinkRunnerContext.get_resource: bind the acquiring
        # action's metric group to the shared cached resource.
        chat_model.set_metric_group(action_group)
        return chat_model

    ctx.get_resource = MagicMock(side_effect=_get_resource)
    return ctx


def test_token_metrics_attributed_to_requesting_action() -> None:
    """Another action rebinding the cached chat model between the request and
    the recording must not steal action A's token metrics.
    """
    chat_model = _SharedChatModelSetup(connection="mock", model="mock-model")
    action_a_group = _MockMetricGroup()
    action_b_group = _MockMetricGroup()

    ctx = _create_ctx(chat_model, action_a_group)

    def _execute_and_interleave(fn: Any, *args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        # Simulate action B acquiring the same cached resource while action
        # A's chat is in flight (mailbox interleaving with async chat).
        chat_model.set_metric_group(action_b_group)
        return result

    ctx.durable_execute = MagicMock(side_effect=_execute_and_interleave)

    asyncio.run(chat(uuid4(), "mock-model", [], None, None, ctx))

    a_model_group = action_a_group.get_sub_group("model", "mock-model")
    assert a_model_group.get_counter("promptTokens").get_count() == 100
    assert a_model_group.get_counter("completionTokens").get_count() == 50
    # Nothing may leak into action B's group.
    assert action_b_group._sub_groups == {}
