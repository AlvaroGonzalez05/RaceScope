import os
import pytest

# ---------------------------------------------------------------------------
# Torch / Metal availability
# ---------------------------------------------------------------------------
# On macOS Apple Silicon, PyTorch's Metal (MPS) framework initialises at the
# first `import torch` call.  In non-interactive subprocess contexts (CI, the
# Claude Code Bash tool, etc.) Metal lacks WindowServer entitlements and the
# initialisation call blocks indefinitely — no signal or timeout can interrupt
# it.  Tests marked `requires_torch` are therefore skipped automatically when
# the `PYTEST_NO_TORCH` environment variable is set to "1", which happens when
# running outside an interactive terminal.  To run these tests, use:
#
#   pytest tests/test_transformer_model.py          # from your own terminal
#
# To skip them explicitly (e.g. in CI):
#
#   PYTEST_NO_TORCH=1 pytest tests/
#
_NO_TORCH = os.environ.get("PYTEST_NO_TORCH", "0") == "1"


def pytest_collection_modifyitems(config, items):
    if _NO_TORCH:
        skip = pytest.mark.skip(reason="PYTEST_NO_TORCH=1 — run from interactive terminal")
        for item in items:
            if item.get_closest_marker("requires_torch"):
                item.add_marker(skip)


# Configure pytest-asyncio to auto mode so all async tests work without decorator
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
