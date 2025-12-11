from typing import Callable, Dict, List
from enum import Enum, auto

class EventType(Enum):
    SIMULATION_START = auto()
    SIMULATION_PAUSE = auto()
    SIMULATION_RESET = auto()
    CREATURE_BORN = auto()
    CREATURE_DIED = auto()
    FOOD_EATEN = auto()
    FOOD_SPAWNED = auto()

class Event:
    def __init__(self, event_type: EventType, data: dict = None):
        self.event_type = event_type
        self.data = data or {}

class EventManager:
    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}

    def subscribe(self, event_type: EventType, callback: Callable):
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable):
        if event_type in self._listeners:
            self._listeners[event_type].remove(callback)

    def emit(self, event: Event):
        if event.event_type in self._listeners:
            for callback in self._listeners[event.event_type]:
                callback(event)
