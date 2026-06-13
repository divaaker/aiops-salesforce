import os
import sys

# Make `import voxbox` work without an install step.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
