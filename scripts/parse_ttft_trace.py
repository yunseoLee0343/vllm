#!/usr/bin/env python3
"""Parse TTFT_TRACE logs and print compact per-request timing table."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

TOKEN_RE = re.compile(r"(\w+)=([^\s]+)")


@dataclass
class Event:
    stage: str
    request_id: str
    parent_request_id: str | None
    t: float
    pid: int
    wall_ns: int


STAGES = {
    "api_ingress",
    "api_request_id_map",
    "api_before_engine_handoff",
    "frontend_add_request_entry",
    "frontend_before_enginecore_add",
    "engine_preprocess",
    "engine_before_scheduler_add",
    "scheduler_waiting_enqueue",
    "scheduler_first_progress",
    "output_first_emit",
}


def parse_line(line: str) -> Event | None:
    if "[TTFT_TRACE]" not in line:
        return None
    fields = dict(TOKEN_RE.findall(line))
    stage = fields.get("stage")
    req_id = fields.get("request_id")
    if not stage or not req_id or stage not in STAGES:
        return None
    try:
        return Event(
            stage=stage,
            request_id=req_id,
            parent_request_id=fields.get("parent_request_id"),
            t=float(fields["t"]),
            pid=int(fields["pid"]),
            wall_ns=int(fields["wall_ns"]),
        )
    except (KeyError, ValueError):
        return None


def choose_delta_ms(start: Event | None, end: Event | None) -> str:
    if start is None or end is None:
        return "-"
    if start.pid == end.pid:
        return f"{(end.t - start.t) * 1000:.3f}(t)"
    return f"{(end.wall_ns - start.wall_ns) / 1e6:.3f}(wall)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="+", help="Log files to parse")
    args = parser.parse_args()

    by_req: dict[str, dict[str, Event]] = {}
    parent_to_children: dict[str, set[str]] = {}

    for path in args.log:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                event = parse_line(line)
                if event is None:
                    continue
                stages = by_req.setdefault(event.request_id, {})
                prev = stages.get(event.stage)
                if prev is None or event.wall_ns < prev.wall_ns:
                    stages[event.stage] = event
                if event.parent_request_id and event.parent_request_id != event.request_id:
                    parent_to_children.setdefault(event.parent_request_id, set()).add(
                        event.request_id
                    )

    print(
        "request_id\tstall_wait_to_progress_ms\tprogress_to_first_emit_ms"
        "\te2e_to_first_emit_ms(best_effort)"
    )
    for req_id in sorted(by_req):
        stages = by_req[req_id]
        stall = choose_delta_ms(
            stages.get("scheduler_waiting_enqueue"),
            stages.get("scheduler_first_progress"),
        )
        emit = choose_delta_ms(
            stages.get("scheduler_first_progress"),
            stages.get("output_first_emit"),
        )

        ingress = stages.get("api_ingress")
        if ingress is None:
            # find parent ingress if this request came from mapped parent
            for parent_id, children in parent_to_children.items():
                if req_id in children:
                    ingress = by_req.get(parent_id, {}).get("api_ingress")
                    if ingress is not None:
                        break

        e2e = choose_delta_ms(ingress, stages.get("output_first_emit"))
        print(f"{req_id}\t{stall}\t{emit}\t{e2e}")


if __name__ == "__main__":
    main()
