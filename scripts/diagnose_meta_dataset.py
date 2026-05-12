#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from scripts.diagnose_metadrive_dataset_quality import main
if __name__=='__main__': main()
