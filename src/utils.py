from typing import Any, Generator, Generic, TypeVar

TYield = TypeVar("TYield")
TReturn = TypeVar("TReturn")


class StatefulGenerator(Generic[TYield, TReturn]):
    def __init__(self, g: Generator[TYield, Any, TReturn]):
        self.g = g
        self.ret: TReturn = None  # pyright: ignore

    def __iter__(self):
        self.ret = yield from self.g

    def consume(self):
        for _ in self:
            pass
        return self


def consume_generator(g: Generator):
    for _ in g:
        pass
