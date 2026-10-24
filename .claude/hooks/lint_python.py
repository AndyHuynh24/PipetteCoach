#!/usr/bin/env python3
"""PostToolUse hook: format + lint the edited Python file with ruff.
If lint errors remain, exit 2 so Claude sees them and fixes them."""

import json
import shutil
import subprocess
import sys

payload = json.load(sys.stdin)
path = (payload.get("tool_input") or {}).get("file_path", "")

if not path.endswith(".py") or shutil.which("ruff") is None:
    sys.exit(0)

subprocess.run(["ruff", "format", path], capture_output=True)
result = subprocess.run(["ruff", "check", "--fix", path], capture_output=True, text=True)

if result.returncode != 0:
    print(f"ruff found issues in {path}:\n{result.stdout}", file=sys.stderr)
    sys.exit(2)
sys.exit(0)
