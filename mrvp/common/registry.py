from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str, value: Callable[..., Any] | None = None):
        def inner(obj: Callable[..., Any]):
            if key in self._items:
                raise KeyError(f"duplicate registry key {key} in {self.name}")
            self._items[key] = obj
            return obj
        if value is None:
            return inner
        return inner(value)

    def get(self, key: str) -> Callable[..., Any]:
        if key not in self._items:
            known = ",".join(sorted(self._items))
            raise KeyError(f"unknown key {key} in {self.name}; known={known}")
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items)
