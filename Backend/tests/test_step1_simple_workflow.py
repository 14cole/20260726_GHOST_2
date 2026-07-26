from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Backend"))

from step1_monostatic import discover_jobs, validate_config
from feature_sum import load_body_profile_grim, save_body_grim
from step2_monostatic import discover_jobs as discover_body_jobs


class Step1SimpleWorkflowTests(unittest.TestCase):
    def test_empty_role_is_valid_and_outputs_are_role_separated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "geometries" / "FRD").mkdir(parents=True)
            (root / "geometries" / "OPN").mkdir(parents=True)
            (root / "geometries" / "OPN" / "case.geo").write_text(
                "placeholder", encoding="utf-8"
            )
            jobs = discover_jobs(root, [3.0], ["TM", "TE"])
            self.assertEqual(len(jobs), 2)
            self.assertTrue(all(job["role"] == "OPN" for job in jobs))
            self.assertTrue(
                all(Path(job["output"]).parent == (root / "results" / "OPN").resolve()
                    for job in jobs)
            )
            self.assertTrue((root / "results" / "FRD").is_dir())

    def test_complete_physical_channels_are_required(self):
        with self.assertRaisesRegex(ValueError, "exactly TM and TE"):
            validate_config([3.0], [0.0, 90.0], ["TM"])

    def test_frequency_filename_collisions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "output-name precision"):
            validate_config([1.0001, 1.0002], [0.0], ["TM", "TE"])

    def test_body_jobs_are_flat_and_profile_is_embedded(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "geometries").mkdir()
            (root / "geometries" / "alpha.geo").write_text("placeholder")
            (root / "geometries" / "beta.geo").write_text("placeholder")
            jobs = discover_body_jobs(root)
            self.assertEqual(
                [Path(job["output"]).name for job in jobs],
                ["alpha.grim", "beta.grim"],
            )
            profile = np.asarray([[0.0, -1.0], [0.5, 0.0], [0.0, 1.0]])
            bodies = {
                3.0: {
                    "theta_deg": [0.0, 90.0, 180.0],
                    "amp_vv": [1 + 0j, 2 + 0j, 1 + 0j],
                    "amp_hh": [1j, 2j, 1j],
                }
            }
            path = root / "results" / "body.grim"
            save_body_grim(bodies, str(path), body_profile=profile)
            np.testing.assert_array_equal(load_body_profile_grim(path), profile)


if __name__ == "__main__":
    unittest.main()
