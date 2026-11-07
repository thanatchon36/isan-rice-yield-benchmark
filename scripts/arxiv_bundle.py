"""Build the arXiv source bundle, ``paper/arxiv.tar.gz``, from paper/ as last built.

arXiv compiles the paper itself (pdfLaTeX) and does not run BibTeX, so the bundle carries main.bbl next to
main.tex, the sections, the generated tables and only the figures the paper uses (PDF). The staged copy is
compiled once with tectonic to prove it is self-contained, then tarred. Run ``make paper`` first (it refreshes
main.bbl). Prints the title, the plain-text abstract (for the arXiv form) and a comments line.

Usage: ``python scripts/arxiv_bundle.py`` (or ``make arxiv``).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
STAGE = PAPER / "arxiv"
OUT = PAPER / "arxiv.tar.gz"


def main() -> int:
    tex = [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]
    figs = sorted({m for f in tex for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", f.read_text())})
    files = [PAPER / "main.tex", PAPER / "main.bbl", PAPER / "references.bib", *tex[1:],
             *sorted((PAPER / "tables").glob("*.tex")), *[PAPER / "figures" / f"{f}.pdf" for f in figs]]
    missing = [str(f.relative_to(ROOT)) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for f in files:
        dst = STAGE / f.relative_to(PAPER)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst)
    # the staged copy must compile on its own
    subprocess.run(["tectonic", "-X", "compile", "--keep-logs", "main.tex"], cwd=STAGE, check=True, capture_output=True)
    log = (STAGE / "main.log").read_text(encoding="utf-8", errors="replace")
    pages = re.search(r"Output written on main\.(?:xdv|pdf) \((\d+) pages", log)
    if pages is None:
        raise SystemExit("staged compile left no page count")
    for name in ("main.log", "main.pdf"):
        (STAGE / name).unlink(missing_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        for f in files:
            tar.add(STAGE / f.relative_to(PAPER), arcname=str(f.relative_to(PAPER)))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} kB, {len(files)} files, {len(figs)} figures)")

    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}", (PAPER / "tables" / "macros.tex").read_text()))
    title = re.search(r"\\title\{(.*?)\}\n", main_tex).group(1)
    abstract = re.search(r"\\begin\{abstract\}\s*(?:\\noindent)?\s*(.*?)\\end\{abstract\}", main_tex, re.S).group(1)
    macros["kgrai"] = "kg/rai"
    abstract = re.sub(r"\\(\w+)\{\}", lambda m: macros.get(m.group(1), m.group(0)), abstract)
    abstract = re.sub(r"\\(\w+)\b", lambda m: macros.get(m.group(1), m.group(0)), abstract)
    abstract = (abstract.replace("~", " ").replace("\\%", "%").replace("--", "-")
                .replace("$r=", "r = ").replace("$", ""))
    body = " ".join(abstract.split())
    print(f"\nTitle:\n{title}")
    print(f"\nAbstract ({len(body)} characters; arXiv allows 1,920):\n{body}")
    print(f"\nComments:\n{pages.group(1)} pages, {len(figs)} figures. All results are from a synthetic benchmark. "
          "Code: https://github.com/thanatchon36/isan-rice-yield-benchmark")
    return 0


if __name__ == "__main__":
    sys.exit(main())
