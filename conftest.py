import sys, os

# Ensure the model_basic/ directory is on sys.path so that
# sub-packages (comm, param, attention) are importable by pytest.
sys.path.insert(0, os.path.dirname(__file__))
