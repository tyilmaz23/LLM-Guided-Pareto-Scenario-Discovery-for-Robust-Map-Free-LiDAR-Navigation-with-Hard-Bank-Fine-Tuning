"""
llm_cache.py

Deterministic on-disk cache for LLM outputs.
"""

import os
import json

class LLMCache:
    def __init__(self, path="llm_cache.json"):
        self.path = path
        if os.path.exists(path):
            with open(path, "r") as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

