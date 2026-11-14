#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#

import os


# The file-based CDK imports `unstructured`, which sends a usage ping to
# packages.unstructured.io at import time unless one of these is set.
os.environ.setdefault("SCARF_NO_ANALYTICS", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")

from .source import SourceSFTPBulk  # noqa: E402


__all__ = ["SourceSFTPBulk"]
