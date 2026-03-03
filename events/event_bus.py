"""
In-process event bus with synchronous handler dispatch.

Usage:
    bus = EventBus()
    bus.subscribe(EventType.NEW_SESSION, twin_agent.on_new_session)
    bus.subscribe(EventType.NEW_SESSION, risk_agent.on_new_session)
    bus.publish(NewSessionEvent(athlete_id="A1", session_id="S42", ...))
"""
from __future__ import annotations
import logging
from collections import defaultdict
from typing import Callable

from .event_types import BaseEvent, EventType

logger = logging.getLogger(__name__)

Handler = Callable[[BaseEvent], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)
        logger.debug("Subscribed %s to %s", handler.__qualname__, event_type)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type] = [
            h for h in self._handlers[event_type] if h is not handler
        ]

    def publish(self, event: BaseEvent) -> None:
        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            logger.warning("No handlers registered for %s", event.event_type)
            return
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Handler %s failed for event %s", handler.__qualname__, event.event_id
                )

    def clear(self) -> None:
        self._handlers.clear()
