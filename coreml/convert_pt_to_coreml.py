#!/usr/bin/env python3
"""Convert TorchScript locomotion policies to Core ML.

The .pt files in this directory are expected to be TorchScript modules whose
forward takes one float32 observation tensor shaped [1, input_size] and returns
one action tensor.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


MODEL_SPECS = {
    "go1": {
        "input_size": 48 + 187,
        "output_size": 12,
        "output_name": "go1_velocity_rough",
    },
    "g1": {
        "input_size": 99 + 187,
        "output_size": 29,
        "output_name": "g1_velocity_rough",
    },
}


def make_coreml_friendly_torchscript(model, example):
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, obs):
            return self.wrapped(obs)

    traced = torch.jit.trace(Wrapper(model).eval(), example, strict=False).eval()
    return torch.jit.freeze(traced)


def infer_spec(path: Path, input_size: int | None, output_name: str | None) -> tuple[int, str]:
    stem = path.stem.lower()
    for prefix, spec in MODEL_SPECS.items():
        if stem.startswith(prefix):
            return (
                input_size if input_size is not None else int(spec["input_size"]),
                output_name if output_name is not None else str(spec["output_name"]),
            )
    if input_size is None:
        raise ValueError(f"Cannot infer input size for {path.name}; pass --input-size")
    return input_size, output_name if output_name is not None else path.stem


def convert_one(
    path: Path,
    output_dir: Path,
    input_size: int | None,
    output_name: str | None,
    minimum_ios: str,
    compute_precision: str,
) -> Path:
    import coremltools as ct
    import numpy as np
    import torch

    resolved_input_size, resolved_output_name = infer_spec(path, input_size, output_name)
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()

    example = torch.zeros((1, resolved_input_size), dtype=torch.float32)
    with torch.no_grad():
        output = model(example)
    if hasattr(output, "shape") and output.numel() == 0:
        raise ValueError(f"{path.name} returned an empty tensor for shape {tuple(example.shape)}")

    model = make_coreml_friendly_torchscript(model, example)

    target = getattr(ct.target, f"iOS{minimum_ios.replace('.', '')}", None)
    if target is None:
        raise ValueError(f"Unsupported coremltools iOS target: {minimum_ios}")

    precision = {
        "float16": ct.precision.FLOAT16,
        "float32": ct.precision.FLOAT32,
    }[compute_precision]

    mlmodel = ct.convert(
        model,
        inputs=[ct.TensorType(name="obs", shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="actions")],
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_precision=precision,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{resolved_output_name}.mlpackage"
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    return output_path


def default_inputs(script_dir: Path) -> Iterable[Path]:
    return sorted(script_dir.glob("*.pt"))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="TorchScript .pt files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir.parent / "MujocoAR" / "Resources",
        help="Directory for generated .mlpackage files",
    )
    parser.add_argument("--input-size", type=int, help="Override observation size")
    parser.add_argument("--output-name", help="Override Core ML model basename")
    parser.add_argument("--minimum-ios", default="17", help="Core ML deployment target, e.g. 17")
    parser.add_argument(
        "--compute-precision",
        choices=("float16", "float32"),
        default="float16",
        help="Core ML program weight precision",
    )
    args = parser.parse_args()

    inputs = args.inputs or list(default_inputs(script_dir))
    if not inputs:
        raise SystemExit("No .pt inputs found")

    for input_path in inputs:
        output_path = convert_one(
            input_path.resolve(),
            args.output_dir.resolve(),
            args.input_size,
            args.output_name,
            args.minimum_ios,
            args.compute_precision,
        )
        print(f"{input_path.name} -> {output_path}")


if __name__ == "__main__":
    main()
