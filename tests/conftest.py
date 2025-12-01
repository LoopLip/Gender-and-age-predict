import sys
from pathlib import Path

# ensure project root is importable for tests
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
