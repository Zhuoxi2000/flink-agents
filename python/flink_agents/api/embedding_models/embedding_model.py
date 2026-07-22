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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple, cast

from pydantic import Field
from typing_extensions import override

from flink_agents.api.resource import Resource, ResourceType

if TYPE_CHECKING:
    from flink_agents.api.metric_group import MetricGroup


class BaseEmbeddingModelConnection(Resource, ABC):
    """Base abstract class for text embedding model connection.

    Responsible for managing model service connection configurations.
    Specific implementations can add their own connection parameters like:
    - Service address (base_url) for remote services
    - API key (api_key) for authenticated services
    - Authentication information, timeouts, etc.

    Provides the basic embedding interface for direct communication with model services.

    One connection can be shared in multiple embedding model setup.
    """

    @classmethod
    @override
    def resource_type(cls) -> ResourceType:
        """Return resource type of class."""
        return ResourceType.EMBEDDING_MODEL_CONNECTION

    @abstractmethod
    def embed(
        self, text: str | Sequence[str], **kwargs: Any
    ) -> list[float] | list[list[float]]:
        """Generate embedding vector for a single text input.

        Converts the input text into a high-dimensional vector representation
        suitable for semantic similarity search and retrieval operations.

        Args:
            text: The text string to convert into an embedding vector.
            **kwargs: Additional parameters passed to the embedding model.

        Returns:
            A list of floating-point numbers representing the embedding vector.
            The dimension of the vector depends on the specific embedding model used.
        """

    def embed_with_usage(
        self, text: str | Sequence[str], **kwargs: Any
    ) -> Tuple[list[float] | list[list[float]], Dict[str, Any] | None]:
        """Generate embeddings and report provider token usage alongside them.

        Mirrors how chat connections surface usage on the response
        (``ChatMessage.extra_args``): usage travels on the call path instead
        of being dropped before it reaches the metrics layer (#858).

        Connections that receive usage from their provider should override
        this method; the default keeps existing third-party connections
        working with no usage reported.

        Args:
            text: The text input(s) to convert into embedding vector(s).
            **kwargs: Additional parameters passed to the embedding model.

        Returns:
            A tuple of (embeddings, usage). ``usage`` is ``None`` when the
            provider reports nothing, otherwise a dict with optional keys
            ``model_name``, ``promptTokens`` and ``totalTokens``.
        """
        return self.embed(text, **kwargs), None


class BaseEmbeddingModelSetup(Resource, ABC):
    """Base abstract class for text embedding model setup.

    Responsible for managing embedding model configurations, such as:
    - Connection to embedding model service (connection)
    - Model name (model)

    Provides the basic embedding interface for generating embeddings from text inputs.
    """

    connection: str | BaseEmbeddingModelConnection = Field(
        description="The referenced connection."
    )
    model: str = Field(description="Name of the embedding model to use.")

    @classmethod
    @override
    def resource_type(cls) -> ResourceType:
        """Return resource type of class."""
        return ResourceType.EMBEDDING_MODEL

    @property
    @abstractmethod
    def model_kwargs(self) -> Dict[str, Any]:
        """Return embedding model settings."""

    @override
    def open(self) -> None:
        self.connection = cast(
            "BaseEmbeddingModelConnection",
            self.resource_context.get_resource(
                self.connection, ResourceType.EMBEDDING_MODEL_CONNECTION
            ),
        )

    def _get_connection(self) -> BaseEmbeddingModelConnection:
        if not isinstance(self.connection, BaseEmbeddingModelConnection):
            err_msg = f"Expect BaseEmbeddingModelConnection, but is {self.connection.__class__.__name__}"
            raise TypeError(err_msg)
        return self.connection

    def embed(
        self, text: str | Sequence[str], **kwargs: Any
    ) -> list[float] | list[list[float]]:
        """Generate embedding vector for a single text query.

        Converts the input text into a high-dimensional vector representation
        suitable for semantic similarity search and retrieval operations.

        Token usage reported by the connection is recorded on this setup's
        bound metric group under the same ``model`` dimension used by chat
        models, with ``promptTokens`` and ``totalTokens`` counters (#858).

        Args:
            text: The text string to convert into an embedding vector.
            **kwargs: Additional parameters passed to the embedding model.

        Returns:
            A list of floating-point numbers representing the embedding vector.
            The dimension of the vector depends on the specific embedding model used.
        """
        merged_kwargs = self.model_kwargs.copy()
        merged_kwargs.update(kwargs)
        embeddings, usage = self._get_connection().embed_with_usage(
            text, **merged_kwargs
        )
        if usage:
            self._record_token_metrics(
                usage.get("model_name") or self.model,
                usage.get("promptTokens"),
                usage.get("totalTokens"),
            )
        return embeddings

    def _record_token_metrics(
        self,
        model_name: str,
        prompt_tokens: int | None,
        total_tokens: int | None,
        metric_group: "MetricGroup | None" = None,
    ) -> None:
        """Record embedding token usage metrics for the given model.

        Embedding APIs report input-side usage only, so unlike chat models
        there is no ``completionTokens`` counter.

        Parameters
        ----------
        model_name : str
            The name of the model used
        prompt_tokens : int | None
            The number of prompt (input) tokens, if reported
        total_tokens : int | None
            The total number of tokens, if reported
        metric_group : MetricGroup | None
            The metric group to record into; falls back to the currently
            bound group when not provided.
        """
        if metric_group is None:
            metric_group = self.metric_group
        if metric_group is None:
            return

        model_group = metric_group.get_sub_group("model", model_name)
        if prompt_tokens:
            model_group.get_counter("promptTokens").inc(prompt_tokens)
        if total_tokens:
            model_group.get_counter("totalTokens").inc(total_tokens)
