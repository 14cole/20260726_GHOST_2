from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "3c_add_cavity" / "run_place_3d.py"
spec = importlib.util.spec_from_file_location("_fastener_csv_runner", RUNNER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = RUNNER
spec.loader.exec_module(RUNNER)


class FastenerCsvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pattern = self.root / "rivet.grim"
        self.pattern.touch()
        # Cylinder of radius 0.03 m over solver-axis z.
        # Production profiles are ordered so this yields the outward radial normal.
        self.generatrix = np.asarray([[0.03, 0.1], [0.03, 0.0]])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_header_aliases_and_multiple_rows(self) -> None:
        table = self.root / "rivet.csv"
        table.write_text(
            "x,y,z,x normal,y normal,z normal\n"
            "0.030,0.020,0.000,1,0,0\n"
            "0.000,0.040,0.030,0,0,1\n",
            encoding="utf-8",
        )
        source, rows = RUNNER.read_placement_csv(str(table))
        self.assertEqual(source, table.resolve())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["normal_cad"], (1.0, 0.0, 0.0))

    def test_tolerance_variants_cycle_in_csv_row_order(self) -> None:
        variants = ["small", "nominal", "large"]
        assigned = RUNNER._assigned_entries(variants, 7, "fasteners.csv")
        self.assertEqual(
            assigned,
            ["small", "nominal", "large", "small", "nominal", "large", "small"],
        )
        with self.assertRaisesRegex(SystemExit, "at least one coordinate"):
            RUNNER._assigned_entries(variants, 2, "fasteners.csv")

    def test_validated_points_derive_or_check_optional_normal(self) -> None:
        table = self.root / "rivet.csv"
        table.write_text(
            "x,y,z,nx,ny,nz\n"
            "0.030,0.020,0.000,1,0,0\n"
            "0.000,0.040,0.030,0,0,1\n",
            encoding="utf-8",
        )
        _source, rows = RUNNER.read_placement_csv(str(table))
        with mock.patch.multiple(
            RUNNER,
            SCALE=1.0,
            NORMAL_TOL_DEG=1.0,
            FREQUENCIES_GHZ=[3.0],
        ):
            points = [
                RUNNER._validated_point(
                    self.generatrix,
                    str(self.pattern),
                    row,
                    f"rivet:{row['line']}",
                )[0]
                for row in rows
            ]
        self.assertEqual(len(points), 2)
        np.testing.assert_allclose(points[0]["location"], [0.0, 0.03, 0.02])
        np.testing.assert_allclose(points[0]["aperture_normal"], [0.0, 1.0, 0.0])

    def test_reversed_normal_is_refused(self) -> None:
        table = self.root / "bad.csv"
        table.write_text(
            "x,y,z,nx,ny,nz\n0.030,0.020,0.000,-1,0,0\n",
            encoding="utf-8",
        )
        _source, rows = RUNNER.read_placement_csv(str(table))
        with mock.patch.multiple(
            RUNNER,
            SCALE=1.0,
            NORMAL_TOL_DEG=15.0,
            FREQUENCIES_GHZ=[3.0],
        ):
            with self.assertRaisesRegex(SystemExit, "outward BoR skin normal"):
                RUNNER._validated_point(
                    self.generatrix,
                    str(self.pattern),
                    rows[0],
                    "rivet:2",
                )


if __name__ == "__main__":
    unittest.main()
