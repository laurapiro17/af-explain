"""Hugging Face Spaces entrypoint — re-exports the Streamlit app.

Spaces expects the entry file to be ``app.py`` at the Space root. This
file just calls the package's ``streamlit_app.main``; the actual UI lives
in ``af_explain.app.streamlit_app`` so it can also be run locally.

Before pushing to HF Spaces:
1. Copy a trained checkpoint to ``model.ckpt`` next to this file.
2. Make sure ``requirements.txt`` is up to date (pinned from uv.lock).
3. ``git push hf main`` (after adding the HF remote).
"""

from __future__ import annotations

import os
from pathlib import Path

# Point the streamlit_app at the bundled checkpoint by default.
_DEFAULT_CKPT = Path(__file__).parent / "model.ckpt"
if _DEFAULT_CKPT.exists():
    os.environ.setdefault("AF_EXPLAIN_DEFAULT_CKPT", str(_DEFAULT_CKPT))

from af_explain.app.streamlit_app import main  # noqa: E402

if __name__ == "__main__":
    main()
