#!/usr/bin/env python3
"""PreToolUse hook: block edits to raw data, labels, and the split definition.
Exit code 2 blocks the tool call and sends stderr back to Claude."""

import json
import sys

PROTECTED = ("data/raw/", "labels.json", "configs/splits.json")

payload = json.load(sys.stdin)
path = (payload.get("tool_input") or {}).get("file_path", "")

if any(p in path for p in PROTECTED):
    print(
        f"Blocked: {path} is protected (raw data / labels / split definition). Ask the user before changing it.",
        file=sys.stderr,
    )
    sys.exit(2)
sys.exit(0)
