#!/usr/bin/env python3
"""Stage 3 trainer.

Does not modify agents/qam.py or main.py. Registers qam_region into the
diagnostic loop (same logging as Stage 2, plus AM weights).
"""
import sys

from absl import app

from agents.qam_region import QAMRegionAgent
import main_diag as md

md.agents["qam_region"] = QAMRegionAgent

if __name__ == "__main__":
    if not any(a == "--agent" or a.startswith("--agent=") for a in sys.argv[1:]):
        sys.argv.append("--agent=agents/qam_region.py")
    app.run(md.main)
