"""Well identity <-> canonical plate coordinates. The ONLY place that converts between the two.

Public well id: {"row": "A"-"H", "col": 1-12}; display string "B7".
Canonical plate coordinates: continuous (x, y) in units of well pitch with A1's centre at (0, 0),
x increasing along the row (A1 -> A12) and y down the column (A1 -> H1). H12 is (11, 7).
"""

ROWS = "ABCDEFGH"
N_ROWS, N_COLS = 8, 12


def well(row: str, col: int) -> dict:
    if row not in ROWS or not 1 <= col <= N_COLS:
        raise ValueError(f"invalid well {row}{col}")
    return {"row": row, "col": col}


def well_str(w: dict) -> str:
    return f"{w['row']}{w['col']}"


def parse_well(s: str) -> dict:
    """'B7' -> {"row": "B", "col": 7}."""
    return well(s[0].upper(), int(s[1:]))


def well_center(w: dict) -> tuple[float, float]:
    """Canonical (x, y) of a well centre."""
    return float(w["col"] - 1), float(ROWS.index(w["row"]))


def all_wells() -> list[dict]:
    """All 96 wells in row-major order (A1, A2, ..., H12)."""
    return [well(r, c) for r in ROWS for c in range(1, N_COLS + 1)]


def nearest_well(x: float, y: float, max_dist: float = 0.5) -> dict | None:
    """Well whose centre is nearest to canonical (x, y), or None if farther than max_dist pitch."""
    c, r = round(x), round(y)
    if not (0 <= c < N_COLS and 0 <= r < N_ROWS):
        return None
    if ((x - c) ** 2 + (y - r) ** 2) ** 0.5 > max_dist:
        return None
    return well(ROWS[r], c + 1)


def well_index(w: dict) -> int:
    """Row-major index 0..95 (A1 = 0, A12 = 11, H12 = 95), the order of all_wells() and localizer GRID."""
    return ROWS.index(w["row"]) * N_COLS + (w["col"] - 1)


def index_well(i: int) -> dict:
    return well(ROWS[i // N_COLS], i % N_COLS + 1)
