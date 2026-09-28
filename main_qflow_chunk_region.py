#!/usr/bin/env python3
"""QAM trainer entry point for the action-chunked QFlow time ablation."""

import sys

from absl import app

from agents.qflow_chunk_region import QFlowChunkRegionAgent
import main as trainer

trainer.agents["qflow_chunk_region"] = QFlowChunkRegionAgent

if __name__ == "__main__":
    if not any(a == "--agent" or a.startswith("--agent=") for a in sys.argv[1:]):
        sys.argv.append("--agent=agents/qflow_chunk_region.py")
    app.run(trainer.main)
