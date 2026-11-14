#
# Copyright (c) 2024 Airbyte, Inc., all rights reserved.
#

import pytest
import tiktoken


class _ByteLevelEncoding:
    """In-memory stand-in for a tiktoken encoding: one token per UTF-8 byte."""

    name = "offline-bytes"

    def encode(self, text, allowed_special=frozenset(), disallowed_special="all"):
        return list(text.encode("utf-8"))


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    # Keeps unit tests from fetching BPE files at run time.
    monkeypatch.setattr(tiktoken, "get_encoding", lambda encoding_name: _ByteLevelEncoding())
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda model_name: _ByteLevelEncoding())
