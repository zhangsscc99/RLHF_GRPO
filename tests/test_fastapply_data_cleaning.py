from __future__ import annotations

import asyncio
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_master.applications.fastapply.build import convert, deduplicate, generate, make, parse
from rl_master.applications.fastapply.build.filter import Filter, filter_row
from rl_master.applications.fastapply.data.ast import build_ast
from rl_master.applications.fastapply.data.types import ChangeContext, Hunk, Span
from rl_master.applications.fastapply.data.utils import expand_context_to_all_units, shrink_to_deepest_common_ancestor


def _sample_row():
    old_source = """import os

def price(items):
    return sum(item.amount for item in items)
"""
    new_source = """import os
from decimal import Decimal

def price(items, tax=Decimal("0")):
    subtotal = sum(item.amount for item in items)
    return subtotal + tax
"""
    diff = "".join(
        difflib.unified_diff(
            old_source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            fromfile="a/billing.py",
            tofile="b/billing.py",
        )
    )
    return {
        "id": "tiny-import-function-change",
        "repo": "tiny",
        "file": "billing.py",
        "commit": "abc123",
        "language": "python",
        "old_source": old_source,
        "new_source": new_source,
        "diff": diff,
    }


def test_span_and_hunk_line_mapping():
    span = Span(3, 8)
    assert len(span) == 6
    assert 5 in span
    assert Span(4, 6) in span
    assert span & Span(6, 10) == Span(6, 8)
    assert Span.merge_multiple_spans([Span(1, 2), Span(3, 4), Span(8, 8)]) == [Span(1, 4), Span(8, 8)]

    hunk = Hunk("""@@ -1,3 +1,4 @@
 import os
-def price(items):
-    return sum(item.amount for item in items)
+from decimal import Decimal
+def price(items, tax=Decimal("0")):
+    return sum(item.amount for item in items) + tax""")
    assert hunk.changes["-"] == [2, 3]
    assert hunk.changes["+"] == [2, 3, 4]
    assert not hunk.is_only_addition
    assert not hunk.is_only_deletion


def test_parser_filter_and_ast_context():
    row = _sample_row()
    parsed = parse.parse_row(row)
    assert parsed["changes"]
    assert parsed["changes"][0]["old"].startswith("import os")
    assert "Decimal" in parsed["changes"][0]["new"]

    assert filter_row(parsed, require_import=True)

    ast_tree = build_ast("python", row["new_source"])
    units = ast_tree.get_units_within_top_level_scope()
    hunks = [Hunk(hunk) for hunk in parse.Parser.split_hunks(row["diff"])]
    context = ChangeContext("+", units, hunks)
    expand_context_to_all_units(units, context)
    function_units = [unit for unit in units if unit.type == "function"]
    assert function_units
    assert shrink_to_deepest_common_ancestor(function_units[0], context).type == "function"


def test_end_to_end_cleaning_pipeline():
    row = _sample_row()
    parsed_rows = parse.process([row])
    assert len(parsed_rows) == 1

    generated_rows = asyncio.run(make.process(None, parsed_rows))
    assert generated_rows[0]["update"] == row["new_source"]

    cleaned_rows = generate.process(generated_rows, require_import=True)
    assert len(cleaned_rows) == 1
    assert cleaned_rows[0]["tokens"] > 0

    deduped_rows = deduplicate.deduplicate(cleaned_rows + cleaned_rows)
    assert len(deduped_rows) == 1

    prompt_rows = convert.process(deduped_rows)
    assert prompt_rows[0]["data_source"] == "apply"
    assert "<updated>" in prompt_rows[0]["response"]
    assert prompt_rows[0]["reward_model"]["ground_truth"] == row["new_source"]


if __name__ == "__main__":
    test_span_and_hunk_line_mapping()
    test_parser_filter_and_ast_context()
    test_end_to_end_cleaning_pipeline()
    print("fastapply data cleaning smoke tests passed")
