"""AG-UI event stream for the Elimam UI (no database, no Qdrant: testable alone).

`ElimamEventStream` adds, on top of the standard AG-UI events, what the
frontend renders; `ElimamAGUIAdapter` plugs it into pydantic-ai's AGUIAdapter.
The route that uses them is api/agui.py.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from pydantic_ai.ui import NativeEvent
from pydantic_ai.ui.ag_ui import AGUIAdapter, AGUIEventStream

from api.tool_presenter import describe_call, describe_result
from retrieval.citations import CitationRenderer, CitationStream, Lookup


def _json_or_text(raw: Any) -> Any:
    """A tool result arrives as a JSON string; the presenter wants the value."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


@dataclass
class ElimamEventStream(AGUIEventStream):
    """The standard AG-UI events, plus what the Elimam UI renders.

    - `metadata` on TOOL_CALL_START / TOOL_CALL_RESULT: title, icon, subtitle,
      status, summary, count, duration_ms, preview (api/tool_presenter.py).
    - a CUSTOM `tool_call_ready` event once a call's arguments are complete:
      TOOL_CALL_START fires while arguments are still streaming, so its
      subtitle (built from the arguments) can only arrive here.
    - a CUSTOM `citation` event per resolved citation tag, and the tags in the
      text stream are rewritten to their canonical form (retrieval/citations.py).
    """

    citations: CitationStream = field(default_factory=CitationStream)
    tool_names: dict[str, str] = field(default_factory=dict)
    tool_args: dict[str, str] = field(default_factory=dict)
    tool_started: dict[str, float] = field(default_factory=dict)

    async def handle_event(self, event: NativeEvent) -> AsyncIterator[BaseEvent]:
        async for base in super().handle_event(event):
            for out in self._decorate(base):
                yield out

    def _custom(self, name: str, value: Any) -> CustomEvent:
        return CustomEvent(name=name, value=value, timestamp=self._get_timestamp())

    def _decorate(self, event: BaseEvent):
        if isinstance(event, TextMessageStartEvent):
            # Tags never span messages: start each message with a clean buffer.
            self.citations = CitationStream(self.citations.renderer)
            yield event

        elif isinstance(event, TextMessageContentEvent):
            text, cards = self.citations.feed(event.delta)
            if text:
                yield event.model_copy(update={"delta": text})
            for card in cards:
                yield self._custom("citation", card)

        elif isinstance(event, TextMessageEndEvent):
            tail = self.citations.flush()
            if tail:
                yield TextMessageContentEvent(
                    message_id=event.message_id,
                    delta=tail,
                    timestamp=event.timestamp,
                )
            yield event

        elif isinstance(event, ToolCallStartEvent):
            self.tool_names[event.tool_call_id] = event.tool_call_name
            self.tool_args[event.tool_call_id] = ""
            yield event.model_copy(
                update={"metadata": describe_call(event.tool_call_name, {})}
            )

        elif isinstance(event, ToolCallArgsEvent):
            self.tool_args[event.tool_call_id] = (
                self.tool_args.get(event.tool_call_id, "") + event.delta
            )
            yield event

        elif isinstance(event, ToolCallEndEvent):
            call_id = event.tool_call_id
            args = _json_or_text(self.tool_args.get(call_id) or "{}")
            if not isinstance(args, dict):
                args = {}
            self.tool_started[call_id] = time.monotonic()
            yield self._custom(
                "tool_call_ready",
                {
                    "tool_call_id": call_id,
                    **describe_call(self.tool_names.get(call_id, ""), args),
                },
            )
            yield event

        elif isinstance(event, ToolCallResultEvent):
            call_id = event.tool_call_id
            started = self.tool_started.pop(call_id, None)
            duration_ms = (
                round((time.monotonic() - started) * 1000) if started else None
            )
            yield event.model_copy(
                update={
                    "metadata": describe_result(
                        self.tool_names.get(call_id, ""),
                        _json_or_text(event.content),
                        duration_ms,
                    )
                }
            )

        else:
            yield event


class ElimamAGUIAdapter(AGUIAdapter):
    # Where citation tags are resolved: the corpus (retrieval.by_id) by default.
    # A plain class attribute (not a dataclass field) so tests can swap it.
    citation_lookup: Lookup | None = None

    def build_event_stream(self) -> ElimamEventStream:
        return ElimamEventStream(
            self.run_input,
            citations=CitationStream(CitationRenderer(self.citation_lookup)),
            accept=self.accept,
            ag_ui_version=self.ag_ui_version,
            include_usage=self.include_usage,
            manage_system_prompt=self.manage_system_prompt,
            preserve_file_data=self.preserve_file_data,
        )
