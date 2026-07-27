"""
MYS-439 — class-level guard: fail CI when a prompt reads a preference field
no UI collects.

## Why this exists (class, not instance)

The class this test guards against is: a prompt-assembly module reads a
NAMED preference field (`preferences["budget"]`, `preferences.get("pace")`)
that nothing on the frontend actually collects and sends — the model then
receives (and can act on, or narrate) a field that is always absent. This
ticket was originally scoped against `reader_profile_agent`'s prompt, which
named five such fields. MYS-436 deleted that agent outright, so there is no
live violation today; this test passes cleanly on the current tree.

That does not make the guard pointless (PM ruling, 2026-07-27, MYS-439):
this is a "second occurrence = write the gate" class-level guard (the
sibling read-side of MYS-437's send-side wired-control spec), and its value
is catching the *next* time someone wires a prompt to a field the UI never
collects — not re-proving today's already-fixed instance. Instance fixed !=
class closed.

## What counts as an offense

Every prompt-assembly module (`agents/*.py`, `core/prompts.py`) currently
treats `preferences` as an OPAQUE dict, passed through whole via
`json.dumps(preferences)` — `core/prompts.py::build_composition_prompt`,
`agents/prompts.py::preferences_block`. Neither ever reads a field by name.
An offense is a NEW literal-key read off a variable/parameter named exactly
`preferences`: `preferences["field"]` or `preferences.get("field", ...)`.
Reading the whole dict opaquely (`json.dumps(preferences)`,
`preferences.items()`, `dict(preferences)`) is not an offense — that's the
safe, already-shipped shape.

## SCOPE (deliberately narrow, not exhaustive)

This is an AST scanner bound to the literal receiver name `preferences`,
over a fixed set of prompt-assembly files (`agents/` + `core/prompts.py`).
It does NOT track a differently-named alias (`user_preferences`,
`prefs = preferences; prefs["budget"]`), attribute-style access on a typed
object (`preferences.budget`, e.g. if `TravelPreferences` — currently
unused in production code — were wired back in), or reads outside the
scanned files. Verified by hand as of this PR that no such alias exists
today. Mirrors the same "receiver-name-bound, narrow the claim rather than
over-promise" scope note the fe sibling guard carries
(`storyland-web/tests/regionCitiesGuard.test.ts`, MYS-681).

## Inventory file

`docs/ui_collected_preference_fields.json`: `fields` maps each read field
name to the FE surface that collects it. Empty today (see the file's own
`_readme`). A field found by the scanner but missing from the inventory
fails CI; a field listed in the inventory but no longer found by the
scanner also fails (stale documentation of a read that was removed —
catches the inventory drifting from reality in either direction).
"""

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO_ROOT / "docs" / "ui_collected_preference_fields.json"

# The prompt-assembly surface. NOT core/executor.py, core/session_state.py,
# core/cache.py, etc: those pass the whole `preferences` dict through
# opaquely (`.items()` for cache-key hashing, `state["user:preferences"] =
# preferences`) without ever reading a NAMED field — that's a different
# (and already-guarded, MYS-401) class, not this ticket's.
SCAN_ROOTS = [REPO_ROOT / "agents", REPO_ROOT / "core" / "prompts.py"]


def _iter_py_files():
    for root in SCAN_ROOTS:
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from sorted(root.rglob("*.py"))


class _PreferencesFieldVisitor(ast.NodeVisitor):
    """Collects every literal field name read off a variable/parameter
    literally named `preferences` via `preferences["x"]` or
    `preferences.get("x", ...)`.
    """

    def __init__(self):
        self.fields: set[str] = set()

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Only a READ is an offense candidate. `preferences["derived"] = value`
        # (ast.Store) or `del preferences["derived"]` (ast.Del) are the service
        # WRITING/removing its own key, not reading a field the FE must have
        # collected -- counting those as reads would demand a nonexistent FE
        # surface for a key that never came from the frontend at all (Eng Lead
        # fix-list, MYS-439, Codex P2).
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "preferences"
            and isinstance(node.ctx, ast.Load)
        ):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                self.fields.add(key.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Name)
            and func.value.id == "preferences"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self.fields.add(node.args[0].value)
        self.generic_visit(node)


def find_named_preference_reads(source: str) -> set:
    """Return every literal field name read off a `preferences`-named
    receiver in `source` (see module docstring for exactly which shapes
    count)."""
    tree = ast.parse(source)
    visitor = _PreferencesFieldVisitor()
    visitor.visit(tree)
    return visitor.fields


def _load_inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text())


def test_detector_self_check_catches_every_offense_shape_and_ignores_safe_shapes():
    assert find_named_preference_reads(
        'def f(preferences):\n    return preferences["budget"]\n'
    ) == {"budget"}
    assert find_named_preference_reads(
        "def f(preferences):\n    return preferences.get('preferred_pace')\n"
    ) == {"preferred_pace"}
    assert find_named_preference_reads(
        "def f(preferences):\n    return preferences.get('budget', 'moderate')\n"
    ) == {"budget"}
    assert find_named_preference_reads(
        "def f(preferences):\n"
        "    a = preferences['budget']\n"
        "    b = preferences.get('preferred_pace')\n"
        "    return a, b\n"
    ) == {"budget", "preferred_pace"}

    safe_sources = [
        # a STORE (assignment) or DEL context is the service writing/removing
        # its own key, not reading a field the FE sent -- must not be
        # collected as an offense (Eng Lead fix-list, MYS-439, Codex P2)
        "def f(preferences):\n    preferences['derived'] = 1\n    return preferences\n",
        "def f(preferences):\n    del preferences['derived']\n    return preferences\n",
        # the exact opaque-passthrough shapes already shipped in this repo
        "import json\ndef f(preferences):\n    return json.dumps(preferences)\n",
        "def f(preferences):\n    return dict(preferences)\n",
        "def f(preferences):\n    return sorted(preferences.items())\n",
        # a bare truthiness/emptiness check is not a field read
        "def f(preferences):\n    if not preferences:\n        return ''\n    return 'x'\n",
        # a differently-named receiver is out of scope by design (SCOPE note)
        "def f(user_preferences):\n    return user_preferences.get('budget')\n",
    ]
    for src in safe_sources:
        assert find_named_preference_reads(src) == set(), src


def test_inventory_file_is_valid_json_with_a_fields_map():
    inventory = _load_inventory()
    fields = inventory.get("fields")
    assert isinstance(fields, dict)
    # The _readme's own promise, and the failure message on the drift test
    # below, both claim every entry NAMES THE FE SURFACE that collects the
    # field -- {"budget": null} or {"budget": ""} would pass a bare
    # isinstance(dict) check while telling QA's dead-control sweep nothing.
    # That's this repo's own "copy claims something the code doesn't do"
    # class (MYS-492/MYS-584); enforce the contract the doc states (Eng
    # Lead fix-list, MYS-439, Codex P2).
    for field, fe_surface in fields.items():
        assert isinstance(fe_surface, str) and fe_surface.strip(), (
            f'inventory entry {field!r} must name a non-empty FE surface '
            f'string (got {fe_surface!r}) -- an empty/null value documents '
            'nothing for QA\'s dead-control sweep to diff against'
        )


def test_every_named_preference_field_read_in_the_prompt_assembly_surface_is_inventoried():
    inventory_fields = set(_load_inventory()["fields"].keys())

    found_by_file = {}
    for path in _iter_py_files():
        fields = find_named_preference_reads(path.read_text())
        if fields:
            found_by_file[str(path.relative_to(REPO_ROOT))] = sorted(fields)

    all_found = {f for fields in found_by_file.values() for f in fields}
    undocumented = all_found - inventory_fields
    assert not undocumented, (
        f"prompt assembly reads preference field(s) {sorted(undocumented)} "
        f"not documented in {INVENTORY_PATH.relative_to(REPO_ROOT)} -- add "
        "an entry naming which FE surface actually collects each field "
        f"before this can ship (offending file(s): {found_by_file})"
    )

    stale = inventory_fields - all_found
    assert not stale, (
        f"docs/ui_collected_preference_fields.json documents field(s) "
        f"{sorted(stale)} that the scanner no longer finds anywhere in "
        f"{[str(r.relative_to(REPO_ROOT)) for r in SCAN_ROOTS]} -- either "
        "the read was removed (delete the stale entry) or the scanner's "
        "SCOPE narrowed past a real remaining read (widen it)."
    )
