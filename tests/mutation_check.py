"""Mutation check: verify the test suite actually catches broken implementations.

A passing test suite proves nothing on its own - a test that never fails is
worse than no test, because it looks like coverage. This deliberately breaks
compute_fieldmap.py in six specific ways and asserts that a named test catches
each one.

Run from the repo root:

    ../../env/bin/python -m tests.mutation_check

Two of these mutations SURVIVED when first written, and both were real gaps:

* "drop s= from irfftn" survived because every shape in the round-trip test
  happened to pad to an even final axis. Real-FFT fast lengths are NOT always
  even - next_fast_len(74, real=True) = 75 - and two validated subjects
  (unfErssm011, unfPain002, canvas z=1811 -> 3645) land on an odd one, so the
  guard is load-bearing in production.
* "ignore real_axis" survived because the test only exercised fast_padded_shape
  directly, not the call site in zerofill_padded_shape.

If a mutation starts reporting SKIP, the source moved under it - re-target the
pattern rather than deleting the case.
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "functions" / "compute_fieldmap.py"
PYTEST = str(REPO.parent.parent / "env" / "bin" / "pytest")

MUTATIONS = [
    ("drop s= from irfftn",
     "sfft.irfftn(spectrum, s=(n0, n1, n2), workers=workers)",
     "sfft.irfftn(spectrum, workers=workers)",
     "tests/functions/test_zerofill.py::test_zerofill_returns_the_input_shape"),
    ("ignore real_axis on the last axis",
     "return tuple(fast_padded_shape(n, zerofill, real_axis=(axis == last))",
     "return tuple(fast_padded_shape(n, zerofill, real_axis=False)",
     "tests/functions/test_zerofill.py::test_compute_bz_uses_the_real_table_on_the_last_axis"),
    ("use exact R*N instead of a fast length",
     "return int(sfft.next_fast_len(target, real=real_axis))",
     "return int(target)",
     "tests/functions/test_zerofill.py::test_fast_padded_shape_avoids_the_prime_trap"),
    ("drop the DC term fix-up",
     "            slab[0, 0] = 1.0 / 3.0          # DC term, undetermined at k=0",
     "            pass",
     "tests/functions/test_refactor_invariance.py"),
    ("pad with 0.0 instead of pad_value",
     "                        pad_value, dtype=dtype)",
     "                        0.0, dtype=dtype)",
     "tests/functions/test_zerofill.py::test_zerofill_pads_with_air"),
    ("forget to demote buffer default to None",
     "        buffer = 50 if buffer is None else buffer",
     "        buffer = 0 if buffer is None else buffer",
     "tests/functions/test_zerofill.py::test_default_buffer_is_still_fifty_when_zerofill_absent"),
]


def main():
    pristine = SRC.read_text()
    backup = SRC.with_suffix(".py.mutation_backup")
    backup.write_text(pristine)

    print(f"{'mutation':45}{'result'}")
    print("-" * 66)
    all_caught = True
    try:
        for name, old, new, test in MUTATIONS:
            if old not in pristine:
                print(f"{name:45}SKIP - pattern not found (source moved?)")
                all_caught = False
                continue
            SRC.write_text(pristine.replace(old, new, 1))
            result = subprocess.run([PYTEST, test, "-q", "--no-header", "-x"],
                                    cwd=REPO, capture_output=True, text=True)
            caught = result.returncode != 0
            print(f"{name:45}{'CAUGHT' if caught else '*** SURVIVED ***'}")
            all_caught &= caught
    finally:
        SRC.write_text(pristine)
        backup.unlink(missing_ok=True)

    print("-" * 66)
    print("all mutations caught" if all_caught
          else "SOME MUTATIONS SURVIVED - the tests do not cover them")
    return 0 if all_caught else 1


if __name__ == "__main__":
    sys.exit(main())
