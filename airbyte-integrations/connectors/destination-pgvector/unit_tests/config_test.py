#
# Copyright (c) 2024 Airbyte, Inc., all rights reserved.
#

from unittest.mock import patch

import pytest
from airbyte_cdk.destinations.vector_db_based import embedder
from airbyte_cdk.destinations.vector_db_based.config import (
    AzureOpenAIEmbeddingConfigModel,
    CodeSplitterConfigModel,
    CohereEmbeddingConfigModel,
    FakeEmbeddingConfigModel,
    MarkdownHeaderSplitterConfigModel,
    OpenAICompatibleEmbeddingConfigModel,
    OpenAIEmbeddingConfigModel,
    SeparatorSplitterConfigModel,
)
from airbyte_cdk.destinations.vector_db_based.document_processor import DocumentProcessor
from pydantic import ValidationError

from destination_pgvector.config import ConfigModel


EMBEDDINGS = [
    ({"mode": "openai", "openai_key": "test-key"}, OpenAIEmbeddingConfigModel),
    ({"mode": "cohere", "cohere_key": "test-key"}, CohereEmbeddingConfigModel),
    ({"mode": "fake"}, FakeEmbeddingConfigModel),
    (
        {
            "mode": "azure_openai",
            "openai_key": "test-key",
            "api_base": "https://example.invalid",
            "deployment": "test-deployment",
        },
        AzureOpenAIEmbeddingConfigModel,
    ),
    (
        {"mode": "openai_compatible", "base_url": "http://localhost:8080/v1", "dimensions": 384},
        OpenAICompatibleEmbeddingConfigModel,
    ),
]


@pytest.fixture
def config():
    return {
        "embedding": {"mode": "fake"},
        "processing": {"chunk_size": 1000},
        "indexing": {
            "host": "localhost",
            "database": "test",
            "username": "test",
            "credentials": {"password": "test"},
        },
    }


@pytest.mark.parametrize("embedding,model", EMBEDDINGS)
def test_embedding_configuration(config, embedding, model):
    config["embedding"] = embedding
    parsed = ConfigModel.parse_obj(config)

    assert isinstance(parsed.embedding, model)
    for name, value in embedding.items():
        assert getattr(parsed.embedding, name) == value


@pytest.mark.parametrize(
    "embedding,client,kwargs,dimensions",
    [
        (
            EMBEDDINGS[0][0],
            "OpenAIEmbeddings",
            {"openai_api_key": "test-key", "max_retries": 15, "disallowed_special": ()},
            1536,
        ),
        (
            EMBEDDINGS[1][0],
            "CohereEmbeddings",
            {"cohere_api_key": "test-key", "model": "embed-english-light-v2.0"},
            1024,
        ),
        (EMBEDDINGS[2][0], "FakeEmbeddings", {"size": 1536}, 1536),
        (
            EMBEDDINGS[3][0],
            "OpenAIEmbeddings",
            {
                "openai_api_key": "test-key",
                "chunk_size": 16,
                "max_retries": 15,
                "openai_api_type": "azure",
                "openai_api_version": "2023-05-15",
                "openai_api_base": "https://example.invalid",
                "deployment": "test-deployment",
                "disallowed_special": (),
            },
            1536,
        ),
        (
            EMBEDDINGS[4][0],
            "LocalAIEmbeddings",
            {
                "model": "text-embedding-ada-002",
                "openai_api_key": "dummy-api-key",
                "openai_api_base": "http://localhost:8080/v1",
                "max_retries": 15,
                "disallowed_special": (),
            },
            384,
        ),
    ],
)
def test_embedding_client_configuration(config, embedding, client, kwargs, dimensions):
    config["embedding"] = embedding
    parsed = ConfigModel.parse_obj(config)

    with patch.object(embedder, client) as mocked_client:
        instance = embedder.create_from_config(parsed.embedding, parsed.processing)

    mocked_client.assert_called_once_with(**kwargs)
    assert instance.embedding_dimensions == dimensions
    if embedding["mode"] in ("openai", "azure_openai"):
        assert instance.chunk_size == 1000


@pytest.mark.parametrize(
    "embedding",
    [
        {},
        {"mode": "unknown"},
        {"mode": "openai"},
        {"mode": "cohere"},
        {"mode": "azure_openai"},
        {"mode": "openai_compatible"},
    ],
)
def test_invalid_embedding_configuration(config, embedding):
    config["embedding"] = embedding
    with pytest.raises(ValidationError):
        ConfigModel.parse_obj(config)


def test_processing_defaults(config):
    processing = ConfigModel.parse_obj(config).processing

    assert processing.chunk_size == 1000
    assert processing.chunk_overlap == 0
    assert processing.text_fields == []
    assert processing.metadata_fields == []
    assert processing.field_name_mappings == []
    assert processing.text_splitter is None


@pytest.mark.parametrize(
    "splitter,model",
    [
        (
            {"mode": "separator", "separators": ['" "', '""'], "keep_separator": True},
            SeparatorSplitterConfigModel,
        ),
        ({"mode": "markdown", "split_level": 3}, MarkdownHeaderSplitterConfigModel),
        ({"mode": "code", "language": "python"}, CodeSplitterConfigModel),
    ],
)
def test_processing_configuration(config, splitter, model):
    config["processing"].update(
        chunk_overlap=10,
        text_fields=["body"],
        metadata_fields=["title"],
        field_name_mappings=[{"from_field": "title", "to_field": "heading"}],
        text_splitter=splitter,
    )
    processing = ConfigModel.parse_obj(config).processing

    assert isinstance(processing.text_splitter, model)
    for name, value in splitter.items():
        assert getattr(processing.text_splitter, name) == value
    assert processing.chunk_overlap == 10
    assert processing.text_fields == ["body"]
    assert processing.metadata_fields == ["title"]
    assert processing.field_name_mappings[0].from_field == "title"
    assert processing.field_name_mappings[0].to_field == "heading"
    assert DocumentProcessor.check_config(processing) is None


@pytest.mark.parametrize(
    "splitter",
    [
        {},
        {"mode": "unknown"},
        {"mode": "code"},
        {"mode": "markdown", "split_level": 0},
        {"mode": "markdown", "split_level": 7},
    ],
)
def test_invalid_splitter_configuration(config, splitter):
    config["processing"]["text_splitter"] = splitter
    with pytest.raises(ValidationError):
        ConfigModel.parse_obj(config)


@pytest.mark.parametrize("separator", ["unquoted", "123", "null", "[]"])
def test_invalid_separator_configuration(config, separator):
    config["processing"]["text_splitter"] = {"mode": "separator", "separators": [separator]}
    processing = ConfigModel.parse_obj(config).processing

    assert DocumentProcessor.check_config(processing) == (
        f"Invalid separator: {separator}. Separator needs to be a valid JSON string using double quotes."
    )


def test_processing_requires_chunk_size(config):
    config["processing"] = {}
    with pytest.raises(ValidationError):
        ConfigModel.parse_obj(config)
