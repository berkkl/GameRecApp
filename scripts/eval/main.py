"""Evaluate the production retrieval path using catalog-specific labels."""

import runpy

if __name__ == "__main__":
    runpy.run_module("scripts.eval.eval_recall", run_name="__main__")
