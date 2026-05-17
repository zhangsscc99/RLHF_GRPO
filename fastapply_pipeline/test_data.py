from __future__ import annotations

import json

from .schemas import ApplyExample


def tiny_apply_examples() -> list[ApplyExample]:
    """Two small examples only; real data mining is intentionally not included yet."""
    py_old = '''def fetch_user(user_id, client):
    response = client.get(f"/users/{user_id}")
    return response.json()
'''
    py_new = '''def fetch_user(user_id, client, cache=None):
    if cache is not None and user_id in cache:
        return cache[user_id]
    response = client.get(f"/users/{user_id}")
    data = response.json()
    if cache is not None:
        cache[user_id] = data
    return data
'''
    js_old = '''export function total(items) {
  return items.reduce((sum, item) => sum + item.price, 0);
}
'''
    js_new = '''export function total(items) {
  if (!Array.isArray(items)) return 0;
  return items.reduce((sum, item) => sum + item.price, 0);
}
'''
    return [
        ApplyExample(
            id="py-search-replace-cache",
            language="python",
            old_source=py_old,
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
            new_source=py_new,
            change_kind="search_replace",
        ),
        ApplyExample(
            id="js-structured-add-guard",
            language="javascript",
            old_source=js_old,
            patch=json.dumps({
                "changes": [{"kind": "replace", "old": js_old.rstrip("\n"), "new": js_new.rstrip("\n")}]
            }, ensure_ascii=False),
            new_source=js_new,
            change_kind="structured_changes",
        ),
    ]
