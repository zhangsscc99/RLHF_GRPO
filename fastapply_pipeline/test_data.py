from __future__ import annotations

from .schemas import ApplyExample


def tiny_apply_examples() -> list[ApplyExample]:
    """Two small examples only; real data mining is intentionally not included yet."""
    return [
        ApplyExample(
            id="py-search-replace-cache",
            language="python",
            old_source='''def fetch_user(user_id, client):
    response = client.get(f"/users/{user_id}")
    return response.json()
''',
            patch='''<<<<<<< SEARCH
def fetch_user(user_id, client):
    response = client.get(f"/users/{user_id}")
    return response.json()
=======
def fetch_user(user_id, client, cache=None):
    if cache is not None and user_id in cache:
        return cache[user_id]
    response = client.get(f"/users/{user_id}")
    data = response.json()
    if cache is not None:
        cache[user_id] = data
    return data
>>>>>>> REPLACE''',
            new_source='''def fetch_user(user_id, client, cache=None):
    if cache is not None and user_id in cache:
        return cache[user_id]
    response = client.get(f"/users/{user_id}")
    data = response.json()
    if cache is not None:
        cache[user_id] = data
    return data
''',
            change_kind="search_replace",
        ),
        ApplyExample(
            id="js-structured-add-guard",
            language="javascript",
            old_source='''export function total(items) {
  return items.reduce((sum, item) => sum + item.price, 0);
}
''',
            patch='''{"changes":[{"kind":"replace","old":"export function total(items) {\n  return items.reduce((sum, item) => sum + item.price, 0);\n}","new":"export function total(items) {\n  if (!Array.isArray(items)) return 0;\n  return items.reduce((sum, item) => sum + item.price, 0);\n}"}]}''',
            new_source='''export function total(items) {
  if (!Array.isArray(items)) return 0;
  return items.reduce((sum, item) => sum + item.price, 0);
}
''',
            change_kind="structured_changes",
        ),
    ]
