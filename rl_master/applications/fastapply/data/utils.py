from __future__ import annotations

"""Unit tree traversal helpers from the FastApply data-cleaning screenshots."""

from collections import deque
from typing import Callable, Dict, Optional

from .types import ChangeContext, Span, Unit, UnitState


class DFS:
    """Find the deepest Unit that contains a target span and matches a predicate."""

    def __init__(self, span: Span, predicate: Optional[Callable[[Unit], bool]] = None):
        self._span = span
        self._predicate = predicate

    def __call__(self, unit: Unit) -> Optional[Unit]:
        if self._span not in unit.span:
            return None
        if self._predicate is not None and not self._predicate(unit):
            return None
        best = unit
        for child in unit.children:
            candidate = self(child)
            if candidate is not None and candidate.span in best.span:
                best = candidate
        return best


def bfs_collect_modified_layers(root: Unit, context: ChangeContext) -> Dict[int, list[Unit]]:
    """Collect modified units by depth from a root Unit."""

    queue = deque([(root, 0)])
    layers: Dict[int, list[Unit]] = {}
    while queue:
        node, depth = queue.popleft()
        state = context.visit(node)
        is_modified = state.is_modified if state else False
        if is_modified:
            layers.setdefault(depth, []).append(node)
        for child in node.children:
            queue.append((child, depth + 1))
    return layers


def find_dca_from_layers(layers: Dict[int, list[Unit]]) -> Optional[Unit]:
    """Find deepest common ancestor (DCA) from BFS modified layers."""

    if not layers:
        return None
    dca: Optional[Unit] = None
    for depth in range(max(layers.keys()) + 1):
        units = layers.get(depth, [])
        if len(units) == 1:
            dca = units[0]
        elif len(units) > 1:
            break
    return dca


def shrink_to_deepest_common_ancestor(unit: Unit, context: ChangeContext) -> Optional[Unit]:
    """Shrink a modified Unit to the deepest common ancestor of modified children."""

    state = context.visit(unit)
    if not state or not state.is_modified:
        return None
    if not unit.children:
        return unit
    layers = bfs_collect_modified_layers(unit, context)
    return find_dca_from_layers(layers) or unit


def down_probe_to_deepest_unit(
    unit: Unit,
    context: ChangeContext,
    predicate: Optional[Callable[[Unit], bool]] = None,
) -> list[Unit]:
    """Probe from a modified Unit down to the deepest impacted child Units."""

    state = context.visit(unit)
    if not state or not state.is_modified:
        return []
    modified_spans = state.get_modified_spans() or [unit.span]
    results: Dict[str, Unit] = {}
    for span in modified_spans:
        probe = DFS(span, predicate)
        result = probe(unit)
        if result is not None:
            results[result.id] = result
    return list(results.values())


def _overlapping_child_spans(unit: Unit, spans: list[Span] | None) -> list[Span] | None:
    if spans is None:
        return None
    overlaps: list[Span] = []
    for span in spans:
        intersection = unit.span & span
        if intersection is not None:
            overlaps.append(intersection)
    return overlaps or None


def expand_context_to_all_units(units: list[Unit], context: ChangeContext) -> None:
    """Propagate top-level hunk states to all nested Unit nodes.

    The original data builder first maps hunks onto top-level Units, then expands
    those states down so rules can ask whether imports/functions/statements are
    modified without having to re-run hunk matching at every depth.
    """

    def dfs(unit: Unit) -> None:
        current = context.visit(unit)
        for child in unit.children:
            child_state = context.visit(child)
            if child_state is None:
                if current and current.is_modified:
                    spans = _overlapping_child_spans(child, current.get_modified_spans())
                    context.add(child, UnitState(child, spans))
                else:
                    context.add(child, UnitState(child, None))
            dfs(child)

    for unit in units:
        dfs(unit)


__all__ = [
    "DFS",
    "find_dca_from_layers",
    "shrink_to_deepest_common_ancestor",
    "down_probe_to_deepest_unit",
    "bfs_collect_modified_layers",
    "expand_context_to_all_units",
]
