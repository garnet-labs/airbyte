#
# Copyright (c) 2024 Airbyte, Inc., all rights reserved.
#

from unittest.mock import patch

from airbyte_cdk.destinations.vector_db_based.config import OpenAIEmbeddingConfigModel
from airbyte_cdk.destinations.vector_db_based.embedder import OpenAIEmbedder
from langchain.embeddings.openai import OpenAIEmbeddings


def test_check_reports_invalid_openai_key():
    embedder = OpenAIEmbedder(OpenAIEmbeddingConfigModel(mode="openai", openai_key="invalid-key"), chunk_size=1000)

    with patch.object(OpenAIEmbeddings, "embed_query", side_effect=Exception("Incorrect API key provided: invalid-key")) as embed_query:
        result = embedder.check()

    embed_query.assert_called_once_with("test")
    assert result is not None
    assert "API key" in result


def test_check_passes_when_embedding_succeeds():
    embedder = OpenAIEmbedder(OpenAIEmbeddingConfigModel(mode="openai", openai_key="valid-key"), chunk_size=1000)

    with patch.object(OpenAIEmbeddings, "embed_query", return_value=[0.0] * 1536):
        assert embedder.check() is None
