from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import tifffile
import torch

from fluoroframe.cli import main
from fluoroframe.data import IntensityScale, Movie, clip_and_target, split_movies
from fluoroframe.model import FramePredictor


def make_movie(directory, count, shape, first=1):
    directory.mkdir()
    grid = np.indices(shape)
    for t in range(first, first + count):
        # A bright object translates along x, above an intensity ramp.
        image = np.full(shape, t, dtype=np.uint16)
        image[(grid[-1] - t % shape[-1]) ** 2 + (grid[-2] - shape[-2] // 2) ** 2 <= 2] += 100
        tifffile.imwrite(directory / f"frame{t}.tif", image, photometric="minisblack",
                         metadata={"axes": "YX" if len(shape) == 2 else "ZYX"})
    return directory


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_numeric_order_and_chronological_split(self):
        with TemporaryDirectory() as tmp:
            directory = make_movie(Path(tmp) / "movie", 12, (9, 11), first=7)
            (directory / "._frame7.tif").touch()
            training, validation = split_movies([directory], None, 0.25)
            self.assertEqual(training[0].paths[0].name, "frame7.tif")
            self.assertEqual(training[0].paths[-1].name, "frame15.tif")
            self.assertEqual(validation[0].paths[0].name, "frame16.tif")
            scale = IntensityScale(0, 1)
            region = (slice(0, 1), slice(0, 1))
            clip, target = clip_and_target(validation[0], 2, 2, region, scale, "cpu")
            self.assertEqual(clip.flatten().tolist(), [16.0, 17.0])
            self.assertEqual(target.item(), 18.0)
            with self.assertRaisesRegex(ValueError, "more than once"):
                split_movies([directory], [directory], 0.2)

    def test_missing_time_point_is_reported(self):
        with TemporaryDirectory() as tmp:
            movie = make_movie(Path(tmp) / "movie", 2, (9, 11))
            (movie / "frame2.tif").rename(movie / "frame3.tif")
            with self.assertRaisesRegex(ValueError, "gap"):
                Movie.open(movie)

    def test_model_streaming_and_gradients_on_odd_shapes(self):
        for shape in [(9, 11), (3, 9, 11)]:
            with self.subTest(shape=shape):
                model = FramePredictor(len(shape), width=2)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.fill_(0.025)
                clip = torch.ones(1, 3, 1, *shape, requires_grad=True)
                prediction = model(clip)
                state = None
                with torch.no_grad():
                    for frame in clip.unbind(1):
                        state, skips = model.observe(frame, state)
                    explicit = model.decode(frame, state, skips)
                torch.testing.assert_close(prediction, explicit)
                self.assertEqual(tuple(prediction.shape), (1, 1, *shape))
                prediction.mean().backward()
                self.assertGreater(clip.grad[:, 0].abs().sum().item(), 0)

    def test_train_evaluate_predict_round_trip(self):
        for shape in [(13, 15), (3, 11, 13)]:
            with self.subTest(shape=shape), TemporaryDirectory() as tmp:
                root = Path(tmp)
                train = make_movie(root / "train", 6, shape)
                val = make_movie(root / "val", 5, shape, first=11)
                run = root / "run"
                with redirect_stdout(StringIO()):
                    main(["train", "--train", str(train), "--val", str(val), "--run", str(run),
                          "--epochs", "1", "--steps-per-epoch", "3", "--width", "2",
                          "--histories", "1", "2", "4", "--val-samples", "2", "--device", "cpu"])
                checkpoint = str(run / "best.pt")
                options = ["--checkpoint", checkpoint, "--device", "cpu"]
                report = StringIO()
                with redirect_stdout(report):
                    main(["evaluate", *options, "--movies", str(val), "--histories", "1", "4"])
                metrics = json.loads(report.getvalue())
                self.assertEqual(metrics["target_crops"], 1)
                self.assertEqual(metrics["voxels"], int(np.prod(shape)))
                self.assertEqual(set(metrics["model_mae"]), {"1", "4"})
                actual = tifffile.imread(val / "frame15.tif").astype(np.float32)
                last = tifffile.imread(val / "frame14.tif").astype(np.float32)
                self.assertAlmostEqual(metrics["copy_mae"], float(np.abs(actual - last).mean()), places=4)
                for history in ("1", "4", "all"):
                    output = root / f"next-{history}.tif"
                    tile = [str(max(1, width // 2)) for width in shape]
                    with redirect_stdout(StringIO()):
                        main(["predict", *options, "--movie", str(val), "--through", "frame14.tif",
                              "--history", history, "--tile", *tile, "--output", str(output)])
                    forecast = tifffile.imread(output)
                    self.assertEqual(forecast.shape, shape)
                    self.assertEqual(forecast.dtype, np.float32)
                    self.assertTrue(np.isfinite(forecast).all())
                np.testing.assert_array_equal(tifffile.imread(root / "next-4.tif"),
                                              tifffile.imread(root / "next-all.tif"))
                # A later acquisition must never affect a forecast made through frame 14.
                tifffile.imwrite(val / "frame15.tif", np.zeros(shape, dtype=np.uint16),
                                 photometric="minisblack")
                repeated = root / "repeated.tif"
                with redirect_stdout(StringIO()):
                    main(["predict", *options, "--movie", str(val), "--through", "frame14.tif",
                          "--history", "4", "--tile", *tile, "--output", str(repeated)])
                np.testing.assert_array_equal(tifffile.imread(repeated), tifffile.imread(root / "next-4.tif"))


if __name__ == "__main__":
    unittest.main()
