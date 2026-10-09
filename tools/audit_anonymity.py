#!/usr/bin/env python3
"""Fail if a release tree contains common identity or machine-local leaks."""
from __future__ import annotations
import argparse,re,sys
from pathlib import Path
DEFAULT_PATTERNS=["/"+"root"+"/","/"+"autodl-tmp"+"/",r"[A-Za-z]:[\\/]Users[\\/]",r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"]
EXTS={".py",".yaml",".yml",".toml",".txt",".md",".json",".sh",".env"}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("root",nargs="?",default="."); args=ap.parse_args(); root=Path(args.root); patterns=[re.compile(p,re.I) for p in DEFAULT_PATTERNS]; problems=[]
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXTS or "__pycache__" in p.parts:continue
        try:text=p.read_text(encoding="utf-8")
        except UnicodeDecodeError:continue
        for n,line in enumerate(text.splitlines(),1):
            for pattern in patterns:
                if pattern.search(line):problems.append(f"{p}:{n}: {pattern.pattern}")
    if problems:
        print("Anonymity audit failed:"); print("\n".join(problems)); return 1
    print(f"Anonymity audit passed: {sum(1 for p in root.rglob('*') if p.is_file())} files scanned"); return 0
if __name__=="__main__":sys.exit(main())
