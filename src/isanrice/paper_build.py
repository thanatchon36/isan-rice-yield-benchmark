"""Build paper/paper.pdf from paper/main.tex with tectonic (XeTeX; runs BibTeX itself).

The numbers in the paper come from paper/tables/macros.tex, which `figures.py` writes from results/.
tectonic keeps main.aux next to main.tex; the web reader (paper_web.py) reads the LaTeX numbering
of sections, tables, figures and equations from it, so the two versions number everything alike.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import config as C

PAPER = C.ROOT / "paper"


def build_pdf(paper_dir: Path = PAPER) -> Path:
    if shutil.which("tectonic") is None:
        raise RuntimeError("tectonic not found; install it (e.g. brew install tectonic) to build the paper")
    subprocess.run(["tectonic", "-X", "compile", "--keep-logs", "--keep-intermediates", "main.tex"],
                   check=True, cwd=str(paper_dir))
    shutil.copyfile(paper_dir / "main.pdf", paper_dir / "paper.pdf")
    return paper_dir / "paper.pdf"
