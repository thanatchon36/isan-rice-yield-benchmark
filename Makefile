PY ?= .venv/bin/python

.PHONY: setup simulate experiments quick diagnostics figures map banner poster paper web arxiv dataset model test all clean

setup:          ## create venv and install the package (needs uv)
	uv venv .venv --python 3.12
	uv pip install --python .venv/bin/python -e ".[dev,geo]"

simulate:       ## simulate the main world -> data/raw/ (synthetic, with its true yield components)
	$(PY) -m isanrice.cli simulate

experiments:    ## E1-E7 -> results/ (cached and resumable; about 7 minutes on one CPU core)
	$(PY) -m isanrice.cli experiments

quick:          ## smoke test of E1-E7 with tiny settings -> results_quick/
	$(PY) -m isanrice.cli experiments --quick

diagnostics:    ## post-hoc diagnosis of the Ridge failures -> results/diag_*
	$(PY) -m isanrice.cli diagnostics

figures:        ## figures, LaTeX tables and paper/tables/macros.tex from results/
	$(PY) -m isanrice.cli figures

map:            ## study-area map of the paper (downloads Natural Earth into data/external/)
	$(PY) scripts/thailand_map.py --paper

banner:         ## README banner -> docs/assets/banner.png
	$(PY) scripts/make_banner.py

poster:         ## LinkedIn poster (1440 x 1800) -> docs/assets/linkedin_poster.png (downloads Fira Sans once)
	$(PY) scripts/make_linkedin_poster.py

paper:          ## paper/paper.pdf (tectonic) and the web reader docs/paper/ (pandoc)
	$(PY) -m isanrice.cli paper

web:            ## web reader only -> docs/paper/index.html, served at thanatchon36.github.io/isan-rice-yield-benchmark/paper/
	$(PY) -m isanrice.cli web

arxiv: paper    ## arXiv source bundle -> paper/arxiv.tar.gz
	$(PY) scripts/arxiv_bundle.py

dataset:        ## stage the Hugging Face dataset -> build/hf_dataset/ (UPLOAD=1 also uploads it)
	$(PY) scripts/hf_dataset.py $(if $(UPLOAD),--upload,)

model:          ## refit the final models, export and verify them -> build/hf_model/ (UPLOAD=1 also uploads them)
	$(PY) scripts/hf_model.py $(if $(UPLOAD),--upload,)

test:
	$(PY) -m pytest -q

all: test experiments diagnostics figures map paper

clean:
	rm -rf results_quick build paper/main.pdf paper/main.aux paper/main.log paper/main.out paper/main.blg paper/arxiv paper/arxiv.tar.gz
