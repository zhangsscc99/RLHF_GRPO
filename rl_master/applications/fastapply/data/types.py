from __future__ import annotations

"""Core data structures for the FastApply data-cleaning pipeline.

The 49 WeChat reference screenshots show the original RL-master data builder
using a small set of reusable primitives:

* :class:`Span` represents an inclusive 1-based line range.
* :class:`Hunk` parses one unified-diff hunk and records exact +/- line ids.
* :class:`Unit` wraps an AST/syntax node with a span and a parent/child tree.
* :class:`UnitState` and :class:`ChangeContext` map hunks back onto Units.

This implementation keeps the same public shape while avoiding hard runtime
requirements on tree-sitter/GitPython/HuggingFace datasets so the project smoke
pipeline can run on this server with only the bundled tiny examples.
"""

import hashlib
import re
from dataclasses import dataclass, field
from itertools import groupby
from operator import itemgetter
from typing import Any, Dict, Iterable, Iterator, Literal, Optional, Sequence, Union


@dataclass(frozen=True, order=True)
class Span:
    """Inclusive 1-based line range."""

    start: int = field(metadata={"help": "starting line number (inclusive)"})
    end: int = field(metadata={"help": "ending line number (inclusive)"})

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < 1:
            raise ValueError("line numbers must be positive")
        if self.start > self.end:
            raise ValueError("starting line number must be <= ending line number")

    def __len__(self) -> int:
        return self.end - self.start + 1

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.end + 1))

    def __contains__(self, item: object) -> bool:
        if isinstance(item, int):
            return self.start <= item <= self.end
        if isinstance(item, Span):
            return self.start <= item.start and item.end <= self.end
        return False

    @property
    def lines(self) -> list[int]:
        return list(iter(self))

    def overlaps(self, other: "Span") -> bool:
        return not (self.end < other.start or other.end < self.start)

    def is_adjacent(self, other: "Span") -> bool:
        return self.end + 1 == other.start or other.end + 1 == self.start

    def intersection(self, other: "Span") -> Optional["Span"]:
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if start > end:
            return None
        return Span(start, end)

    def __and__(self, other: "Span") -> Optional["Span"]:
        return self.intersection(other)

    def union(self, other: "Span") -> Union["Span", tuple["Span", "Span"]]:
        if self.overlaps(other) or self.is_adjacent(other):
            return Span(min(self.start, other.start), max(self.end, other.end))
        return (self, other) if self.start <= other.start else (other, self)

    def __or__(self, other: "Span") -> Union["Span", tuple["Span", "Span"]]:
        return self.union(other)

    def subtract(self, other: "Span") -> Union[Optional["Span"], tuple["Span", "Span"]]:
        """Return the part(s) of ``self`` that remain after removing ``other``."""

        if not self.overlaps(other):
            return self
        # other completely covers self
        if other.start <= self.start and self.end <= other.end:
            return None
        # other is strictly in the middle
        if self.start < other.start and other.end < self.end:
            return Span(self.start, other.start - 1), Span(other.end + 1, self.end)
        # overlap on right
        if self.start < other.start <= self.end <= other.end:
            return Span(self.start, other.start - 1)
        # overlap on left
        if other.start <= self.start <= other.end < self.end:
            return Span(other.end + 1, self.end)
        return self

    def __sub__(self, other: "Span") -> Union[Optional["Span"], tuple["Span", "Span"]]:
        return self.subtract(other)

    @classmethod
    def from_lines(cls, lines: Sequence[int]) -> Union["Span", list["Span"]]:
        """Group a non-empty list of line numbers into contiguous Span(s)."""

        if not lines:
            raise ValueError("line number list cannot be empty")
        spans: list[Span] = []
        sorted_lines = sorted(set(int(line) for line in lines))
        for _, group in groupby(enumerate(sorted_lines), key=lambda x: x[0] - x[1]):
            run = [line for _, line in group]
            spans.append(cls(run[0], run[-1]))
        return spans[0] if len(spans) == 1 else spans

    @classmethod
    def merge_multiple_spans(cls, spans: Sequence["Span"]) -> Union["Span", list["Span"]]:
        """Merge overlapping/adjacent spans while preserving disjoint islands."""

        if not spans:
            raise ValueError("span list cannot be empty")
        ordered = sorted(spans, key=lambda span: (span.start, span.end))
        merged: list[Span] = [ordered[0]]
        for current in ordered[1:]:
            last = merged[-1]
            union = last | current
            if isinstance(union, Span):
                merged[-1] = union
            else:
                merged.append(current)
        return merged[0] if len(merged) == 1 else merged


def _normalize_span_list(spans: Union[Span, Sequence[Span], None]) -> Optional[list[Span]]:
    if spans is None:
        return None
    if isinstance(spans, Span):
        return [spans]
    return list(spans)


class Hunk:
    """One unified-diff hunk.

    The hunk header has the common format ``@@ -old_start,old_len
    +new_start,new_len @@``.  The parser records exact removed and added line
    numbers after accounting for context lines inside the hunk body.
    """

    __slots__ = (
        "_header",
        "_body",
        "_changes",
        "_old_start",
        "_old_len",
        "_old_end",
        "_new_start",
        "_new_len",
        "_new_end",
        "_is_only_deletion",
        "_is_only_addition",
    )

    pattern = re.compile(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@")

    def __init__(self, content: str):
        if not content.strip().startswith("@@"):
            raise ValueError("Hunk does not start with @@ indicator")
        lines = content.splitlines()
        if not lines:
            raise ValueError("empty hunk")
        header = lines[0]
        match = self.pattern.search(header)
        if match is None:
            raise ValueError(f"invalid hunk header: {header!r}")
        old_start = int(match.group(1))
        old_len = int(match.group(2) or 1)
        new_start = int(match.group(3))
        new_len = int(match.group(4) or 1)
        changes = self.get_exact_modified_lines(old_start, new_start, lines[1:])

        self._old_start = old_start
        self._old_len = old_len
        self._old_end = old_start + max(old_len, 1) - 1
        self._new_start = new_start
        self._new_len = new_len
        self._new_end = new_start + max(new_len, 1) - 1
        self._changes = changes
        self._is_only_deletion = bool(changes["-"]) and not changes["+"]
        self._is_only_addition = bool(changes["+"]) and not changes["-"]
        self._header = "@@ -{},{} +{},{} @@".format(old_start, old_len, new_start, new_len)
        self._body = "\n".join(lines[1:])

    @staticmethod
    def get_exact_modified_lines(old_start: int, new_start: int, lines: Sequence[str]) -> Dict[Literal["+", "-"], list[int]]:
        deleted_lines: list[int] = []
        added_lines: list[int] = []
        deleted_offset = 0
        added_offset = 0
        for line in lines:
            if line.startswith("\\"):
                continue
            if line.startswith("-"):
                deleted_lines.append(old_start + deleted_offset)
                deleted_offset += 1
            elif line.startswith("+"):
                added_lines.append(new_start + added_offset)
                added_offset += 1
            else:
                deleted_offset += 1
                added_offset += 1
        return {"-": deleted_lines, "+": added_lines}

    @property
    def header(self) -> str:
        return self._header

    @property
    def body(self) -> str:
        return self._body

    @property
    def changes(self) -> Dict[Literal["+", "-"], list[int]]:
        return {"-": list(self._changes["-"]), "+": list(self._changes["+"])}

    @property
    def old_start(self) -> int:
        return self._old_start

    @property
    def old_len(self) -> int:
        return self._old_len

    @property
    def old_end(self) -> int:
        return self._old_end

    @property
    def new_start(self) -> int:
        return self._new_start

    @property
    def new_len(self) -> int:
        return self._new_len

    @property
    def new_end(self) -> int:
        return self._new_end

    @property
    def is_only_deletion(self) -> bool:
        return self._is_only_deletion

    @property
    def is_only_addition(self) -> bool:
        return self._is_only_addition

    def __hash__(self) -> int:
        return hash(self._header + self._body)

    def __repr__(self) -> str:
        return f"Hunk(header={self.header!r}, body={self.body!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Hunk) and self._header == other._header and self._body == other._body


class Unit:
    """AST/syntax unit with source span and parent/children links."""

    __slots__ = ("_node", "_data", "_parent", "_children", "_id")

    def __init__(
        self,
        node: Any = None,
        data: Optional[Dict[str, Any]] = None,
        parent: Optional["Unit"] = None,
        children: Optional[list["Unit"]] = None,
    ):
        self._node = node
        self._data = dict(data or {})
        self._parent = parent
        self._children: list[Unit] = []
        self._id = self._make_id()
        if children:
            for child in children:
                self.add_child(child)

    def _make_id(self) -> str:
        span = self.span
        raw = f"{self.type}:{span.start}:{span.end}:{self._data.get('name', '')}:{id(self._node)}"
        digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
        return f"{self.type}:{span.start}-{span.end}:{digest}"

    def add_child(self, child: "Unit") -> None:
        child._parent = self
        self._children.append(child)

    def get_child(self, index: int) -> Optional["Unit"]:
        if self._children is not None and 0 <= index < len(self._children):
            return self._children[index]
        return None

    @property
    def node(self) -> Any:
        return self._node

    @property
    def children(self) -> list["Unit"]:
        return self._children

    @property
    def parent(self) -> Optional["Unit"]:
        return self._parent

    @property
    def type(self) -> str:
        return str(self._data.get("type", getattr(self._node, "type", "unknown")))

    @property
    def data(self) -> Dict[str, Any]:
        return dict(self._data)

    @property
    def span(self) -> Span:
        raw = self._data.get("span") or self._data.get("_span")
        if isinstance(raw, Span):
            return raw
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            return Span(int(raw[0]), int(raw[1]))
        if self._node is not None and hasattr(self._node, "start_point") and hasattr(self._node, "end_point"):
            start = int(self._node.start_point[0]) + 1
            end = int(self._node.end_point[0]) + 1
            return Span(start, end)
        return Span(1, 1)

    @property
    def id(self) -> str:
        return self._id

    def __hash__(self) -> int:
        return hash(self._id)

    def __repr__(self) -> str:
        parent = "None" if self._parent is None else self._parent.id
        return f"Unit(id={self._id!r}, parent={parent!r}, type={self.type!r}, span={self.span!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Unit) and self._id == other._id


class UnitState:
    """Modification state for a Unit inside one diff operation (+ or -)."""

    __slots__ = ("_unit", "_modified_spans", "_is_modified", "_is_partially_modified")

    def __init__(self, unit: Unit, modified_spans: Union[Span, Sequence[Span], None] = None):
        self._unit = unit
        spans = _normalize_span_list(modified_spans)
        self._modified_spans = spans
        self._is_modified = spans is not None
        is_partially_modified = False
        if self._is_modified and spans:
            is_partially_modified = any(span.start != unit.span.start or span.end != unit.span.end for span in spans)
        self._is_partially_modified = is_partially_modified

    def __repr__(self) -> str:
        return f"UnitState(unit={self._unit.id!r}, modified_spans={self._modified_spans!r})"

    @property
    def unit(self) -> Unit:
        return self._unit

    @property
    def is_modified(self) -> bool:
        return self._is_modified

    @property
    def is_partially_modified(self) -> bool:
        return self._is_partially_modified

    def get_modified_spans(self) -> Optional[list[Span]]:
        if self._modified_spans is None:
            return None
        return list(self._modified_spans)


class ChangeContext:
    """Map diff hunks to UnitState objects for old (-) or new (+) source."""

    __slots__ = ("_operation", "_states")

    def __init__(self, operation: Literal["+", "-"], units: Sequence[Unit], hunks: Sequence[Hunk]):
        self._operation = operation
        self._states: Dict[str, UnitState] = self.build(operation, units, hunks)

    @staticmethod
    def build(operation: Literal["+", "-"], units: Sequence[Unit], hunks: Sequence[Hunk]) -> Dict[str, UnitState]:
        mapping: Dict[str, list[int]] = {unit.id: [] for unit in units}
        for unit in units:
            for hunk in hunks:
                for line in hunk.changes[operation]:
                    if line in unit.span:
                        mapping[unit.id].append(line)
        states: Dict[str, UnitState] = {}
        for unit in units:
            lines = sorted(set(mapping.get(unit.id, [])))
            spans: Optional[Union[Span, list[Span]]]
            spans = Span.from_lines(lines) if lines else None
            states[unit.id] = UnitState(unit, spans)
        return states

    @property
    def operation(self) -> Literal["+", "-"]:
        return self._operation

    def visit(self, unit: Unit) -> Optional[UnitState]:
        return self._states.get(unit.id)

    def add(self, unit: Unit, state: UnitState) -> None:
        if unit.id not in self._states:
            self._states[unit.id] = state
        elif self._states[unit.id] is not state:
            # The original screenshot keeps context immutable after insertion.
            # Do the same to catch accidental conflicting state propagation.
            raise RuntimeError("unit and its change state cannot be modified after initialization")

    def states(self) -> Dict[str, UnitState]:
        return dict(self._states)

    def modified_units(self) -> list[Unit]:
        return [state.unit for state in self._states.values() if state.is_modified]


__all__ = ["Span", "Hunk", "Unit", "UnitState", "ChangeContext"]
