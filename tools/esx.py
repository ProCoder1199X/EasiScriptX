#!/usr/bin/env python3
import argparse
import sys

def main():
    p = argparse.ArgumentParser(description="Run ESX script without compiling (mock interpreter).")
    p.add_argument("script", help="Path to .esx file")
    p.add_argument("--print", dest="do_print", action="store_true", help="Print parsed actions")
    args = p.parse_args()

    try:
        with open(args.script, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Failed to read {args.script}: {e}", file=sys.stderr)
        return 1

    # Minimal sandboxed preview: only allows a subset of tokens, rejects suspicious chars
    forbidden = ["`", "$(`", "|", ";;", "&&", "||", "<(", ">("]
    if any(tok in content for tok in forbidden):
        print("Rejected script: contains forbidden shell-like constructs", file=sys.stderr)
        return 2

    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if args.do_print:
        for i, ln in enumerate(lines, 1):
            print(f"{i:04d}: {ln}")
    else:
        # Mock execution: just acknowledge
        print(f"Parsed {len(lines)} non-empty lines from {args.script}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


