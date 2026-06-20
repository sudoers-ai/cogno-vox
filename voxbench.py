#!/usr/bin/env python3
"""Root shim — run the vox round-trip bench with `python3 voxbench.py`.

Examples:
    python3 voxbench.py --stub                 # deterministic smoke (no network)
    python3 voxbench.py                         # real STT/TTS via env config
    python3 voxbench.py --stub --max-wer 0.0    # CI gate
"""

import sys

from cognobench.runner import main

if __name__ == "__main__":
    sys.exit(main())
