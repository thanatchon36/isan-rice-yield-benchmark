"""Build an arXiv-ready source archive (arxiv_source.zip) from paper/.
arXiv does not run BibTeX, so paper/main.bbl is included; only the figures used by the paper are packed.
Rebuild main.bbl after changing references:  cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
"""
import re
import zipfile
from pathlib import Path

P = Path(__file__).resolve().parent / "paper"
tex = [P / "main.tex", *sorted((P / "sections").glob("*.tex"))]
figs = sorted({m for f in tex for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", f.read_text())})
files = [P / "main.tex", P / "main.bbl", P / "references.bib", *sorted((P / "sections").glob("*.tex")),
         *sorted((P / "tables").glob("*.tex")), *[P / "figures" / f"{f}.pdf" for f in figs]]
missing = [str(f) for f in files if not f.exists()]
assert not missing, f"missing: {missing}"
with zipfile.ZipFile("arxiv_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(P))
print(f"arxiv_source.zip: {len(files)} files ({len(figs)} figures)")
