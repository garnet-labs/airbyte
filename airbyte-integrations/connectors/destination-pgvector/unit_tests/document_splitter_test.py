#
# Copyright (c) 2024 Airbyte, Inc., all rights reserved.
#

from airbyte_cdk.destinations.vector_db_based.config import ProcessingConfigModel
from airbyte_cdk.destinations.vector_db_based.document_processor import DocumentProcessor
from airbyte_cdk.models import (
    AirbyteRecordMessage,
    AirbyteStream,
    ConfiguredAirbyteCatalog,
    ConfiguredAirbyteStream,
    DestinationSyncMode,
    SyncMode,
)


def _catalog() -> ConfiguredAirbyteCatalog:
    stream = AirbyteStream(name="documents", json_schema={}, supported_sync_modes=[SyncMode.full_refresh])
    return ConfiguredAirbyteCatalog(
        streams=[
            ConfiguredAirbyteStream(
                stream=stream,
                sync_mode=SyncMode.full_refresh,
                destination_sync_mode=DestinationSyncMode.append,
            )
        ]
    )


def _record(text: str) -> AirbyteRecordMessage:
    return AirbyteRecordMessage(stream="documents", data={"id": 1, "str_col": text, "category": "a"}, emitted_at=0)


def test_long_text_is_split_into_multiple_chunks():
    config = ProcessingConfigModel.parse_obj({"chunk_size": 20, "text_fields": ["str_col"], "metadata_fields": ["category"]})
    processor = DocumentProcessor(config, _catalog())

    chunks, id_to_delete = processor.process(_record("Airbyte moves data from sources to destinations. " * 20))

    assert len(chunks) > 1
    assert id_to_delete is None
    for chunk in chunks:
        assert chunk.page_content
        assert chunk.metadata["category"] == "a"


def test_short_text_stays_in_one_chunk():
    config = ProcessingConfigModel.parse_obj({"chunk_size": 1000, "text_fields": ["str_col"], "metadata_fields": []})
    processor = DocumentProcessor(config, _catalog())

    chunks, _ = processor.process(_record("short text"))

    assert len(chunks) == 1
    assert "short text" in chunks[0].page_content
