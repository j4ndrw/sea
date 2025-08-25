import readability
from bs4 import BeautifulSoup
from typing import Any, Callable, Generator, Generic, TypeVar

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


def html_to_text(html: str) -> str:
    doc = readability.Document(html)
    parsed_html = doc.summary()

    soup = BeautifulSoup(parsed_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    return text

T = TypeVar('T')
R = TypeVar('R')

def distinct_by(map_fn: Callable[[T], R], pool: list[T]) -> list[T]:
    seen = set()
    distinct_items = []

    for item in pool:
        value = map_fn(item)
        if value not in seen:
            seen.add(value)
            distinct_items.append(item)

    return distinct_items
