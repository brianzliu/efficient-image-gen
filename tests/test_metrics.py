import unittest
import numpy as np
from PIL import Image
from evaluation.metrics import similarity, integrate_energy


class MetricsTest(unittest.TestCase):
    def test_energy_interpolates_window_boundaries(self):
        self.assertAlmostEqual(integrate_energy([(0, 10), (1, 20), (2, 30)], .5, 1.5), 20)

    def test_energy_refuses_extrapolation_and_gaps(self):
        for samples, start, end in [([(0,10),(1,10)],0,2), ([(0,10),(5,10)],1,4)]:
            with self.assertRaises(ValueError):
                integrate_energy(samples, start, end)

    def test_energy_refuses_duplicate_time(self):
        with self.assertRaises(ValueError):
            integrate_energy([(0,10),(0,10)],0,1)

    def test_identity_and_known_pixel_error(self):
        a = Image.fromarray(np.full((16,16,3),100,dtype=np.uint8))
        b = Image.fromarray(np.full((16,16,3),110,dtype=np.uint8))
        self.assertEqual(similarity(a,a)["ssim"],1)
        self.assertTrue(similarity(a,a)["identical_pixels"])
        self.assertAlmostEqual(similarity(a,b)["psnr_db"], 20*np.log10(25.5))

    def test_mismatched_sizes_fail(self):
        with self.assertRaises(ValueError):
            similarity(Image.new("RGB", (16,16)), Image.new("RGB",(32,32)))


if __name__ == "__main__":
    unittest.main()
