"""Test safety defaults: never load or download the multi-GB STT model during tests."""
import os

os.environ.setdefault("SAYSO_DISABLE_MODEL_AUTOLOAD", "1")
