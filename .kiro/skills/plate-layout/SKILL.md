---
name: plate-layout
description: Plate layout JSON schema and well-naming rules for 96-well plates. Use whenever writing code or prompts that create, read, validate, or display plate layouts or well IDs.
---

# Plate layout format

Wells: rows A–H, columns 1–12. A1 = top-left when the plate is in standard orientation.
Well ID in JSON: `{"row": "B", "col": 7}`. String form for display/logs: `"B7"` (no zero padding).

Layout JSON (validated by `protocol/schemas/layout.schema.json`):

```json
{
  "name": "Bradford assay",
  "plate_type": "96",
  "steps": [
    {
      "step": 1,
      "reagent": "BSA standard 2 mg/mL",
      "volume_ul": 10,
      "wells": ["A1", "A2", "A3"],
      "channel_mode": "single"
    },
    {
      "step": 2,
      "reagent": "Sample 1",
      "volume_ul": 10,
      "wells": ["B1", "B2"],
      "channel_mode": "single"
    }
  ]
}
```

Rules:
- `channel_mode`: `"single"`, `"multi8"` (one column, A–H), or `"multi12"` (one row, 1–12).
  For multi modes, `wells` lists every well filled in that stroke.
- A well may appear in several steps (e.g., sample then reagent) — that is not a double-fill.
  Double-fill = same well twice within the SAME step.
- Run-time errors to flag: `wrong_well`, `skipped_well`, `double_fill`, `out_of_order` (warning only).
- When generating layouts with Claude, always validate against the schema and show the layout
  on the grid for user confirmation before a run starts.
