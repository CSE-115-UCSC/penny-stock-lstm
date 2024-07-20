from dataclasses import dataclass
from typing import List
from .Point import Point

@dataclass
class Sequence:
    ticker: str
    date: str
    points: List[Point]