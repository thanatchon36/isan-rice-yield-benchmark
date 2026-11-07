"""isan-rice-yield-benchmark command line: python -m isanrice.cli <command> (or `isanrice <command>`)."""
from __future__ import annotations

import argparse
import sys


def cmd_simulate(args, rest):
    from . import config as C
    from .simulate import simulate

    t = simulate(seed=args.seed, scenario=args.scenario, out_dir=C.DATA_RAW)
    print(f"simulated world (seed {args.seed}, {args.scenario}) -> {C.DATA_RAW}:",
          {k: v.shape for k, v in t.items() if hasattr(v, "shape")})


def cmd_experiments(args, rest):
    from .experiments import main

    main(rest)


def cmd_diagnostics(args, rest):
    from . import diag_worlds, diagnostics

    diagnostics.main()
    diag_worlds.main()


def cmd_figures(args, rest):
    from .figures import main

    main()


def cmd_paper(args, rest):
    from .paper_build import build_pdf
    from .paper_web import build_web

    print("wrote", build_pdf())
    print("wrote", build_web())


def cmd_web(args, rest):
    from .paper_web import build_web

    print("wrote", build_web())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="isanrice", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("simulate", help="simulate the main world -> data/raw/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario", default="base")
    p.set_defaults(fn=cmd_simulate)
    sub.add_parser("experiments", help="E1-E7 -> results/ (cached; extra args: --only e3 --force | --quick)",
                   add_help=False).set_defaults(fn=cmd_experiments)
    sub.add_parser("diagnostics", help="post-hoc Ridge diagnostics -> results/diag_*").set_defaults(fn=cmd_diagnostics)
    sub.add_parser("figures", help="figures, LaTeX tables and macros -> paper/").set_defaults(fn=cmd_figures)
    sub.add_parser("paper", help="paper/paper.pdf (tectonic) and the web reader docs/paper/").set_defaults(fn=cmd_paper)
    sub.add_parser("web", help="web reader only -> docs/paper/index.html (pandoc)").set_defaults(fn=cmd_web)
    args, rest = ap.parse_known_args(argv)
    if rest and args.command != "experiments":
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    args.fn(args, rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
