"""Command-line interface for binary vessel connectivity repair."""
import argparse
import json
from pathlib import Path

from .demo import DEFAULT_OUTDIR, run_demo
from .io_nifti import affine_in_mm, load_binary, save_binary
from .repair import RepairParameters, repair_mask


def build_parser() -> argparse.ArgumentParser:
    """Build the documented command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--tube-radius-mm", type=float, default=1.0)
    parser.add_argument("--max-added-fg-fraction", type=float, default=0.05)
    parser.add_argument("--min-component-size", type=int, default=1)
    parser.add_argument("--max-gap-mm", type=float, default=20.0)
    parser.add_argument("--filter-island-mm", type=float, default=0.0)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--connectivity", type=int, choices=[26], default=26)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Repair an input NIfTI or run the synthetic demo and print JSON metrics."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.demo and (args.input or args.output):
        parser.error("--demo cannot be combined with --input or --output")
    if not args.demo and (args.input is None or args.output is None):
        parser.error("Provide both --input and --output, or use --demo")
    p = RepairParameters(**{name: getattr(args, name) for name in RepairParameters.__dataclass_fields__})
    try:
        p.validate()
        if args.demo:
            summary = run_demo(args.outdir, p)
        else:
            if args.input.resolve() == args.output.resolve():
                raise ValueError("Use distinct input and output paths to preserve the source")
            mask, reference = load_binary(args.input)
            result = repair_mask(mask, affine_in_mm(reference), p)
            save_binary(result.mask, reference, args.output)
            summary = {"input_path": str(args.input.resolve()),
                       "output_path": str(args.output.resolve()), **result.summary()}
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
