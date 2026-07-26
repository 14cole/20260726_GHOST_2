from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_bias import conservative_occluder


class ShadowBiasTests(unittest.TestCase):
    def setUp(self):
        self.triangles = np.asarray([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ])
        self.points = np.asarray([[2.0, 2.0, 0.0]])
        self.normals = np.asarray([[0.0, 0.0, 1.0]])
        self.directions = np.asarray([[0.0, 0.0, 1.0]])

    def test_stable_visibility_selects_half_default(self):
        occluder, info = conservative_occluder(
            self.triangles,
            scale=1.0,
            points=self.points,
            normals=self.normals,
            directions=self.directions,
        )
        self.assertEqual(info["mode"], "automatic_conservative")
        self.assertTrue(info["half_default_stable"])
        self.assertAlmostEqual(
            occluder.bias, 0.5 * info["mesh_default_bias_m"]
        )

    def test_explicit_override_is_audited(self):
        occluder, info = conservative_occluder(
            self.triangles,
            scale=1.0,
            points=self.points,
            normals=self.normals,
            directions=self.directions,
            override_m=1e-5,
        )
        self.assertEqual(info["mode"], "advanced_override")
        self.assertEqual(occluder.bias, 1e-5)


if __name__ == "__main__":
    unittest.main()
