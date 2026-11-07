"""Reproduce everything: tests -> experiments E1-E7 -> post-hoc diagnostics -> figures/tables -> paper.pdf"""
import subprocess
import sys

STEPS = [([sys.executable, "-m", "pytest", "-q", "tests"], None), ([sys.executable, "experiments.py"], None),
         ([sys.executable, "diagnostics.py"], None), ([sys.executable, "diag_worlds.py"], None),
         ([sys.executable, "figures.py"], None),
         ([sys.executable, "thailand_map.py", "--paper"], None), (["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], "paper")]
for cmd, cwd in STEPS:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)
