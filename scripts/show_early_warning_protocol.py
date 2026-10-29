"""Print the frozen Phase 0 early-warning protocol as JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dews.experiment_config import DEFAULT_EARLY_WARNING_CONFIG


def main() -> int:
    print(json.dumps(DEFAULT_EARLY_WARNING_CONFIG.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
