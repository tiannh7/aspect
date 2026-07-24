#!/usr/bin/env python3

"""Run and validate the focused native center-of-mass test matrix."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REFERENCE_MASS = 12.0 * math.pi
OUTER_RADIUS = 2.0
RELATIVE_TOLERANCE = 1.0e-2
Y20_TRANSLATION_TOLERANCE = 1.0e-8
MPI_RELATIVE_TOLERANCE = 1.0e-8
MPI_ABSOLUTE_TOLERANCE = 1.0e-9

_DIAGNOSTIC = re.compile(
    r"Coupled center-of-mass constraint:.*?"
    r"dipole pre \[kg m\]=\(([^)]*)\), "
    r"translation \[m\]=\(([^)]*)\), "
    r"dipole residual \[kg m\]=\(([^)]*)\)"
)


@dataclass(frozen=True)
class Diagnostic:
    dipole_pre: tuple[float, float, float]
    translation: tuple[float, float, float]
    dipole_residual: tuple[float, float, float]

    @property
    def relative_residual(self) -> float:
        scale = max(
            _norm(self.dipole_pre),
            100.0
            * sys.float_info.epsilon
            * REFERENCE_MASS
            * OUTER_RADIUS,
        )
        return _norm(self.dipole_residual) / scale


def _vector(text: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in text.split(","))
    if len(values) != 3:
        raise ValueError(f"expected three vector components, got {text!r}")
    return values  # type: ignore[return-value]


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _diagnostics(output: str, case_name: str) -> list[Diagnostic]:
    result = [
        Diagnostic(
            _vector(match.group(1)),
            _vector(match.group(2)),
            _vector(match.group(3)),
        )
        for match in _DIAGNOSTIC.finditer(output)
    ]
    if not result:
        raise RuntimeError(
            f"{case_name}: no coupled center-of-mass diagnostic was found"
        )
    for diagnostic in result:
        values = (
            *diagnostic.dipole_pre,
            *diagnostic.translation,
            *diagnostic.dipole_residual,
            diagnostic.relative_residual,
        )
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{case_name}: non-finite COM diagnostic")
    return result


def _execute(
    name: str,
    command: Sequence[str],
    working_directory: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    environment.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    completed = subprocess.run(
        command,
        cwd=working_directory,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (working_directory / f"{name}.log").write_text(completed.stdout)
    return completed


def _run_case(
    name: str,
    command: Sequence[str],
    working_directory: Path,
) -> list[Diagnostic]:
    completed = _execute(name, command, working_directory)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name}: ASPECT exited with {completed.returncode}; "
            f"see {working_directory / f'{name}.log'}"
        )

    diagnostics = _diagnostics(completed.stdout, name)
    final_residual = diagnostics[-1].relative_residual
    if final_residual > RELATIVE_TOLERANCE:
        raise RuntimeError(
            f"{name}: final relative D-Mc residual {final_residual:.6e} "
            f"exceeds {RELATIVE_TOLERANCE:.6e}"
        )
    print(
        f"{name:20s} iterations={len(diagnostics):2d} "
        f"relative_residual={final_residual:.6e} "
        f"translation={_norm(diagnostics[-1].translation):.6e} m"
    )
    return diagnostics


def _run_expected_failure(
    name: str,
    command: Sequence[str],
    working_directory: Path,
    expected_message: str,
) -> None:
    completed = _execute(name, command, working_directory)
    if completed.returncode == 0:
        raise RuntimeError(f"{name}: expected failure but ASPECT succeeded")
    normalized_stdout = " ".join(completed.stdout.split())
    normalized_message = " ".join(expected_message.split())
    if normalized_message not in normalized_stdout:
        raise RuntimeError(
            f"{name}: failed for an unexpected reason; see "
            f"{working_directory / f'{name}.log'}"
        )
    print(f"{name:20s} expected failure confirmed")


def _resolve_command(command: str) -> Path:
    candidate = Path(command).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise ValueError(f"command does not exist: {resolved}")
        return resolved

    located = shutil.which(command)
    if located is None:
        raise ValueError(f"command is not in PATH: {command}")
    return Path(located).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aspect",
        required=True,
        type=Path,
        help="Path to aspect-release or aspect-debug",
    )
    parser.add_argument(
        "--mpirun",
        help="Optional mpirun path or command name for the MPI consistency gate",
    )
    parser.add_argument("--mpi-ranks", type=int, default=2)
    parser.add_argument(
        "--keep-output",
        type=Path,
        help="Keep logs and model output in this directory",
    )
    arguments = parser.parse_args()

    aspect = arguments.aspect.expanduser().resolve()
    if not aspect.is_file():
        parser.error(f"ASPECT executable does not exist: {aspect}")
    if arguments.mpi_ranks < 1:
        parser.error("--mpi-ranks must be positive")

    source_root = Path(__file__).resolve().parents[1]
    cases = {
        "native-y10": source_root / "tests/potential_feedback_center_of_mass.prm",
        "relaxed-y10": source_root
        / "tests/potential_feedback_center_of_mass_relaxed.prm",
        "fixed-inner": source_root
        / "tests/potential_feedback_center_of_mass_fixed_inner.prm",
        "native-y20": source_root
        / "tests/potential_feedback_center_of_mass_y20.prm",
        "missing-free-cmb": source_root
        / "tests/potential_feedback_center_of_mass_missing_cmb.prm",
    }

    remove_output_after_success = arguments.keep_output is None
    if remove_output_after_success:
        working_directory = Path(
            tempfile.mkdtemp(prefix="aspect-center-of-mass-")
        )
    else:
        working_directory = arguments.keep_output.expanduser().resolve()
        working_directory.mkdir(parents=True, exist_ok=True)

    succeeded = False
    try:
        results: dict[str, list[Diagnostic]] = {}
        for name in ("native-y10", "relaxed-y10", "fixed-inner", "native-y20"):
            results[name] = _run_case(
                name,
                [str(aspect), str(cases[name])],
                working_directory,
            )

        _run_expected_failure(
            "missing-free-cmb",
            [str(aspect), str(cases["missing-free-cmb"])],
            working_directory,
            "CMB mass feedback unless the bottom boundary has zero normal velocity",
        )

        relaxed = results["relaxed-y10"]
        if len(relaxed) < 2:
            raise RuntimeError(
                "relaxed-y10: expected more than one potential iteration"
            )
        peak_relaxed_residual = max(
            diagnostic.relative_residual for diagnostic in relaxed
        )
        if relaxed[-1].relative_residual >= peak_relaxed_residual:
            raise RuntimeError(
                "relaxed-y10: D-Mc residual did not decrease from its peak"
            )

        y20_translation = _norm(results["native-y20"][-1].translation)
        if y20_translation > Y20_TRANSLATION_TOLERANCE:
            raise RuntimeError(
                f"native-y20: translation {y20_translation:.6e} m exceeds "
                f"{Y20_TRANSLATION_TOLERANCE:.6e} m"
            )

        if arguments.mpirun is not None:
            try:
                mpirun = _resolve_command(arguments.mpirun)
            except ValueError as error:
                parser.error(str(error))
            parallel = _run_case(
                f"native-y10-mpi{arguments.mpi_ranks}",
                [
                    str(mpirun),
                    "-np",
                    str(arguments.mpi_ranks),
                    str(aspect),
                    str(cases["native-y10"]),
                ],
                working_directory,
            )
            serial_translation = results["native-y10"][-1].translation
            parallel_translation = parallel[-1].translation
            difference = _norm(
                tuple(
                    serial - distributed
                    for serial, distributed in zip(
                        serial_translation, parallel_translation
                    )
                )
            )
            scale = max(_norm(serial_translation), sys.float_info.min)
            relative_difference = difference / scale
            if (relative_difference > MPI_RELATIVE_TOLERANCE
                and difference > MPI_ABSOLUTE_TOLERANCE):
                raise RuntimeError(
                    "MPI consistency gate failed: translation difference "
                    f"{difference:.6e} m exceeds absolute tolerance "
                    f"{MPI_ABSOLUTE_TOLERANCE:.6e} m and relative "
                    f"difference={relative_difference:.6e} exceeds "
                    f"{MPI_RELATIVE_TOLERANCE:.6e}"
                )

        succeeded = True
        print("All coupled center-of-mass validation gates passed.")
        if arguments.keep_output is not None:
            print(f"Logs retained in {working_directory}")
        return 0
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print(f"Logs retained in {working_directory}", file=sys.stderr)
        return 1
    finally:
        if remove_output_after_success and succeeded:
            shutil.rmtree(working_directory)


if __name__ == "__main__":
    raise SystemExit(main())
