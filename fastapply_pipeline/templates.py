from __future__ import annotations

SYSTEM_PROMPT = "You are a coding assistant specialized in merging code updates."

USER_TEMPLATE = """Please apply all changes from <patch> to <source>.
- Preserve code structure, order, comments, and indentation exactly unless the patch requires changes.
- Do not include explanations, placeholders, ellipses, markdown fences, or conflict markers.
- Output only the updated source code within <updated> and </updated> tags.

<language>{language}</language>
<source>
{source}
</source>
<patch>
{patch}
</patch>
"""


def build_prompt(language: str, source: str, patch: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n" + USER_TEMPLATE.format(
        language=language.strip(),
        source=source.strip("\n"),
        patch=patch.strip("\n"),
    )


def build_response(updated_source: str) -> str:
    return "<updated>\n" + updated_source.strip("\n") + "\n</updated>"
