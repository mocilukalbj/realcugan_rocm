#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Real-CUGAN Async V3 Launcher - 使用 Python 启动避免编码问题"""
import subprocess
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
# 使用 venv 的 Python，而不是当前解释器
PYTHON = SCRIPT_DIR / "venv" / "Scripts" / "python.exe"

def main():
    print("=" * 50)
    print("  Real-CUGAN Async V3")
    print("=" * 50)
    print()

    script = SCRIPT_DIR / "run_video_async.py"
    if not script.exists():
        print(f"[Error] Script not found: {script}")
        sys.exit(1)

    CONFIG = SCRIPT_DIR / "配置_批处理.txt"
    if not CONFIG.exists():
        print(f"[Error] Config not found: {CONFIG}")
        sys.exit(1)

    print(f"[Python] {PYTHON}")
    print(f"[Script] {script}")
    print(f"[Config] {CONFIG}")
    print()

    try:
        with open(CONFIG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("输入") or line.startswith("输出"):
                    print(f"  {line}")
    except Exception as e:
        print(f"[Warn] Cannot read config: {e}")

    print()
    print("Starting...")
    print()

    cmd = [str(PYTHON), "-u", str(script), "--config", str(CONFIG)]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()