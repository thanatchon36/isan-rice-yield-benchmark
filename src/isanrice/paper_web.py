"""Build the web reader docs/paper/ from paper/main.tex (same text and numbers as the PDF).

pandoc renders the LaTeX source with paper/reader.html: masthead, abstract, a results strip whose numbers
come from paper/tables/macros.tex (the file the PDF uses), contents, numbered sections, references and the
appendices. Section, table, figure and equation numbers are taken from paper/main.aux, so they match the
PDF; run `make paper` first. The PDF and the figures it uses are copied next to the page; GitHub Pages
serves docs/ at https://thanatchon36.github.io/isan-rice-yield-benchmark/. Requires pandoc on PATH.
"""
from __future__ import annotations

import datetime as dt
import html
import re
import shutil
import subprocess
from pathlib import Path

from . import config as C

REPO_URL = "https://github.com/thanatchon36/isan-rice-yield-benchmark"
PAPER_URL = "https://thanatchon36.github.io/isan-rice-yield-benchmark/paper/"
DATA_URL = "https://huggingface.co/datasets/thanatchon36/isan-rice-yield-benchmark"
MODEL_URL = "https://huggingface.co/thanatchon36/isan-rice-yield-benchmark"
CONCEPT_DOI = "10.5281/zenodo.24036674"
PAPER = C.ROOT / "paper"
OUT = C.ROOT / "docs" / "paper"

# (label, style, finding, detail) with {macro} placeholders from paper/tables/macros.tex.
# style: "ref" = result on the synthetic benchmark, "null" = pending.
RESULTS = [
    ("Accuracy", "ref",
     "LightGBM RMSE {rmseLGB} kg/rai against {rmseClim} for climatology",
     "validation-selected LightGBM + Ridge ensemble {rmseMain}; the gain replicated in 10 independent worlds"),
    ("Leakage", "ref",
     "Random cross-validation understates the error by {gainRandomLGB} to {gainRandom}%",
     "and hides the catastrophic seasons of the linear learner, traced post hoc to row-wise LOO tuning"),
    ("Satellite", "ref",
     "Dropping satellite features changes LightGBM RMSE by {ablLGBSatellite}%",
     "the only feature group whose removal matters; its value appears from October "
     "(November RMSE {leadNov} with, {leadNoSatNov} without)"),
    ("Explanations", "ref",
     "Grouped SHAP recovers drought (r = {recDrought})",
     "but misattributes heat stress (r = {recHeat}) and absorbs the technology trend into prices"),
    ("Uncertainty", "ref",
     "80% conformal intervals cover {covEighty}% of outcomes",
     "rolling calibration on past seasons; the worst season, {covEightyMinYear}, covers {covEightyMin}%"),
    ("Real data", "null",
     "Empirical results pending",
     "every number here comes from the synthetic benchmark; real inputs in data/raw/ and make all replace them"),
]


def _macros(paper_dir: Path) -> dict[str, str]:
    text = "".join((paper_dir / "tables" / f).read_text(encoding="utf-8") for f in ("macros.tex", "macros_map.tex"))
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}", text))


def _labels(paper_dir: Path) -> dict[str, str]:
    aux = paper_dir / "main.aux"
    if not aux.exists():
        raise RuntimeError("paper/main.aux not found; run `make paper` (tectonic) before `make web`")
    return dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux.read_text(encoding="utf-8", errors="replace")))


def _latex_title(tex: str) -> tuple[str, str]:
    title = re.search(r"\\title\{(.*?)\}\n", tex).group(1)
    head, _, sub = title.partition(": ")
    return head, sub[:1].upper() + sub[1:]


def _keywords(tex: str) -> tuple[str, list[str]]:
    """Remove the keywords paragraph from the body and return it as a list (masthead metadata)."""
    m = re.search(r"^\\noindent\\textbf\{\\textit\{Keywords\}\}\\enspace (.*)$", tex, re.M)
    if m is None:
        return tex, []
    words = [w.strip() for w in m.group(1).split(r"\textperiodcentered{}")]
    return tex[:m.start()] + tex[m.end():], words


def _renumber(page: str, labels: dict[str, str]) -> str:
    """LaTeX numbering everywhere: cross-references, headings (appendix A, B, ...), TOC and captions."""
    def ref(m: re.Match) -> str:
        return m.group(1) + labels.get(m.group(2), m.group(3)) + "</a>"
    page = re.sub(r'(<a [^>]*data-reference="([^"]+)"[^>]*>)([^<]*)</a>', ref, page)

    # pandoc numbers the appendices 8, 9, ...; LaTeX letters them. Map each top-level number that a
    # labelled heading shows a letter for, then rewrite every heading/TOC number under it.
    top = {}
    for m in re.finditer(r'<h1 data-number="(\d+)" id="([^"]+)"', page):
        lab = labels.get(m.group(2), m.group(1))
        if lab != m.group(1) and lab.isalpha():
            top[m.group(1)] = lab

    def num(n: str) -> str:
        first, dot, rest = n.partition(".")
        return top.get(first, first) + dot + rest
    page = re.sub(r'(data-number=")([\d.]+)(")', lambda m: m.group(1) + num(m.group(2)) + m.group(3), page)
    page = re.sub(r'(<span class="(?:header|toc)-section-number">)([\d.]+)(</span>)',
                  lambda m: m.group(1) + num(m.group(2)) + m.group(3), page)

    def figure(m: re.Match) -> str:
        n = labels.get(m.group(2))
        return m.group(0) + (f"<b>Figure {n}.</b> " if n else "")
    page = re.sub(r'(<figure id="([^"]+)"[^>]*>(?:(?!</figure>).)*?<figcaption>)', figure, page, flags=re.S)

    def table(m: re.Match) -> str:
        n = labels.get(m.group(2))
        return m.group(0) + (f"<b>Table {n}.</b> " if n else "")
    return re.sub(r'(<div id="(tab:[^"]+)">\s*<table[^>]*>\s*<caption>)', table, page)


def _web_images(page: str) -> str:
    """LaTeX names the figures without extension (\\graphicspath); the web page uses the PNG renders."""
    def img(m: re.Match) -> str:
        tag = re.sub(r'\s(?:style|width|height)="[^"]*"', "", m.group(0))
        tag = re.sub(r'src="(?:figures/)?([\w-]+)(?:\.pdf|\.png)?"', r'src="figures/\1.png"', tag)
        return tag.replace("<img ", '<img loading="lazy" ', 1)
    return re.sub(r"<img\b[^>]*>", img, page)


def _references_before_appendix(page: str, labels: dict[str, str]) -> str:
    """citeproc appends the bibliography after the appendices; the PDF has it before them."""
    start, end = page.find('<div id="refs"'), page.rfind("</article>")
    app = re.search(r'<h1 data-number="[A-Z]"', page)
    if start < 0 or app is None or app.start() > start:
        return page
    refs = page[start:end].rstrip()
    page = page[:start] + page[end:]
    heading = '<h1 class="unnumbered" id="references">References</h1>\n'
    page = page[:app.start()] + heading + refs + "\n" + page[app.start():]
    toc_app = re.search(r'<li><a href="#[^"]*" id="toc-[^"]*"><span class="toc-section-number">[A-Z]</span>', page)
    if toc_app:
        page = page[:toc_app.start()] + '<li><a href="#references">References</a></li>\n' + page[toc_app.start():]
    return page


def build_web(paper_dir: Path = PAPER, out_dir: Path = OUT) -> Path:
    """main.tex -> out_dir/index.html (pandoc, reader.html), plus paper.pdf and the figures it shows."""
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc not found; install pandoc to build the web reader")
    labels, macros = _labels(paper_dir), _macros(paper_dir)
    tex = (paper_dir / "main.tex").read_text(encoding="utf-8")
    title, subtitle = _latex_title(tex)
    tex, keywords = _keywords(tex)
    date = dt.date.fromtimestamp((paper_dir / "paper.pdf").stat().st_mtime) if (paper_dir / "paper.pdf").exists() \
        else dt.date.today()
    args = [
        "pandoc", "--from=latex", "--to=html5", "--mathml", "--wrap=none", "--standalone",
        "--template=reader.html", "--number-sections", "--toc", "--toc-depth=1",
        "--citeproc", "--bibliography=references.bib",
        f"--metadata=title:{title}", f"--metadata=subtitle:{subtitle}",
        "--metadata=author:Thanatchon Chongmankhong", f"--metadata=date:Preprint · {date:%-d %B %Y}",
        "--metadata=pagetitle:Isan rice yield benchmark",
        ("--metadata=description:Preprint: leakage-safe, explainable machine learning for district-level rice "
         "yield forecasting, a ground-truth simulation benchmark for Northeast Thailand."),
        *[f"--metadata=keywords:{k}" for k in keywords],
    ]
    page = subprocess.run(args, input=tex, capture_output=True, text=True, check=True, cwd=str(paper_dir)).stdout

    page = _renumber(page, labels)
    page = _web_images(page)
    page = _references_before_appendix(page, labels)
    # The caption sits above the table, outside its sideways scroll, so it wraps at the text width.
    page = re.sub(r"<table\b[^>]*>\s*<caption>(.*?)</caption>(.*?)</table>",
                  r'<div class="tbl"><p class="tcap">\1</p><div class="tscroll"><table>\2</table></div></div>',
                  page, flags=re.DOTALL)
    # Column widths are sized for the PDF page; in the browser the tables size to their content and scroll.
    page = re.sub(r"<colgroup>.*?</colgroup>\s*", "", page, flags=re.DOTALL)
    unresolved = sorted(set(re.findall(r'data-reference="[^"]+"[^>]*>\[[^<]*\]', page)))
    if unresolved:
        raise RuntimeError(f"unresolved cross-references: {unresolved}")

    items = "\n".join(
        f'<li class="r r-{style}"><span class="r-id">{html.escape(label)}</span>'
        f'<span class="r-text">{html.escape(finding.format_map(macros))}'
        f'<small>{html.escape(detail.format_map(macros))}</small></span></li>'
        for label, style, finding, detail in RESULTS
    )
    actions = (
        '<li><a href="paper.pdf">Download PDF</a></li>'
        f'<li><a href="{REPO_URL}">Code and replication</a></li>'
        f'<li><a href="{DATA_URL}">Data (Hugging Face)</a></li>'
        f'<li><a href="{MODEL_URL}">Model (Hugging Face)</a></li>'
        f'<li><a href="https://doi.org/{CONCEPT_DOI}">Zenodo archive</a></li>'
    )
    bib = (
        "@misc{chongmankhong2026isanrice,\n"
        "  author = {Chongmankhong, Thanatchon},\n"
        f"  title  = {{{title}: {subtitle[:1].lower() + subtitle[1:]}}},\n"
        "  year   = {2026},\n"
        f"  doi    = {{{CONCEPT_DOI}}},\n"
        f"  url    = {{{PAPER_URL}}},\n"
        "  note   = {Preprint; all results are from a synthetic benchmark}\n"
        "}"
    )
    cite = f"<h2>How to cite</h2><pre><code>{html.escape(bib)}</code></pre>"
    colophon = (
        "<p>Rendered from <code>paper/main.tex</code> by <code>isanrice.paper_web</code> (<code>make web</code>). "
        "Every number comes from <code>paper/tables/macros.tex</code>, which <code>make figures</code> writes from "
        "the result files in <code>results/</code>; numbering follows the LaTeX build, and the PDF is "
        "authoritative. All data are synthetic.</p>"
    )
    for marker, value in (
        ("<!--ACTIONS-->", actions),
        ("<!--RESULTS-NOTE-->", (
            "Headline numbers, generated from the result files. All of them come from the synthetic benchmark "
            "and describe forecasting pipelines, not Thai rice yields. Amber: benchmark results; grey: pending."
        )),
        ("<!--RESULTS-->", items),
        ("<!--CITE-->", cite),
        ("<!--COLOPHON-->", colophon),
    ):
        if marker not in page:
            raise RuntimeError(f"reader.html lacks {marker}")
        page = page.replace(marker, value)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    pdf = paper_dir / "paper.pdf"
    if pdf.exists():
        shutil.copyfile(pdf, out_dir / "paper.pdf")
    if (out_dir / "figures").exists():
        shutil.rmtree(out_dir / "figures")
    (out_dir / "figures").mkdir()
    for name in sorted(set(re.findall(r'src="figures/([\w.-]+)"', page))):
        shutil.copyfile(paper_dir / "figures" / name, out_dir / "figures" / name)
    return out_dir / "index.html"
