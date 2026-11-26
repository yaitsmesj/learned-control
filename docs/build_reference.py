"""Paste the tested files into the Reference, so doc and code cannot drift.

Run from the repo root:  python docs/build_reference.py
"""
# [permanent practice] Generate docs from the source of truth instead of copying by hand.
import re
from pathlib import Path

src = Path("docs/reference_src.md").read_text()


def paste(m):
    path, lang = m.group(1), m.group(2)
    body = Path(path).read_text().rstrip("\n")
    fence = "````" if "```" in body else "```"
    return f"**`{path}`**\n\n{fence}{lang}\n{body}\n{fence}"


out = re.sub(r"<<<FILE (\S+) (\S+)>>>", paste, src)
Path("The Learned Control Reference.md").write_text(out)
print("wrote The Learned Control Reference.md,", out.count("\n"), "lines")
