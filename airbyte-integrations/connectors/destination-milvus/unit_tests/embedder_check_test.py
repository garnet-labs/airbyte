#
# Copyright (c) 2024 Airbyte, Inc., all rights reserved.
#

from airbyte_cdk.destinations.vector_db_based.config import OpenAIEmbeddingConfigModel
from airbyte_cdk.destinations.vector_db_based.embedder import OpenAIEmbedder


def test_check_reports_invalid_openai_key():
    embedder = OpenAIEmbedder(OpenAIEmbeddingConfigModel(mode="openai", openai_key="invalid-key"), chunk_size=1000)

    result = embedder.check()

    assert result is not None
    assert "API key" in result
