# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from collections.abc import Mapping
from typing import Any


def is_ttft_trace_enabled() -> bool:
    return os.getenv("VLLM_TRACE_TTFT", "0") == "1"


def log_ttft_trace(
    logger,
    *,
    stage: str,
    request_id: str,
    parent_request_id: str | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> None:
    """Emit one machine-friendly TTFT trace log line.

    Field order is stable to simplify parsing:
    stage, request_id, parent_request_id (if applicable), t, pid, wall_ns,
    then any caller-provided extra fields.

    Trace values should remain whitespace-free (`key=value` tokens) so simple
    whitespace-based parsers can consume logs reliably.
    """
    fields = [f"[TTFT_TRACE] stage={stage}", f"request_id={request_id}"]
    if parent_request_id is not None:
        fields.append(f"parent_request_id={parent_request_id}")
    fields.append(f"t={time.perf_counter():.6f}")
    fields.append(f"pid={os.getpid()}")
    fields.append(f"wall_ns={time.time_ns()}")
    if extra_fields:
        for key, value in sorted(extra_fields.items()):
            fields.append(f"{key}={value}")
    logger.info(" ".join(fields))
