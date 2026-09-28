#!/usr/bin/env python3
"""Trainer entry point for the target-weighted QAM screen."""

import sys

from absl import app

from agents.qam_target_region import QAMTargetRegionAgent
import main_diag as md

md.agents["qam_target_region"] = QAMTargetRegionAgent

if __name__ == "__main__":
    if not any(a == "--agent" or a.startswith("--agent=") for a in sys.argv[1:]):
        sys.argv.append("--agent=agents/qam_target_region.py")
    app.run(md.main)
