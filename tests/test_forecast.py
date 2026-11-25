from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import tifffile
import torch

from fluoroframe.cli import main
from fluoroframe.batching import configure_fp32, largest_passing_batch, probe_cuda_batch
from fluoroframe.data import IntensityScale, Movie, clip_and_target, split_movies, training_batch
from fluoroframe.experiment import load_checkpoint
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

    def test_batch_probe_finds_last_safe_integer(self):
        for limit in (1, 3, 8, 19):
            with self.subTest(limit=limit):
                self.assertEqual(largest_passing_batch(lambda b: b <= limit), limit)
        with self.assertRaisesRegex(ValueError, "batch 1"):
            largest_passing_batch(lambda b: False)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_probe_accepts_default_device(self):
        device = torch.device("cuda")
        configure_fp32(device)

        def single_trial(fits):
            self.assertTrue(fits(1))
            return 1

        with patch("fluoroframe.batching.largest_passing_batch", side_effect=single_trial):
            size, trials = probe_cuda_batch(device, (3, 9, 11), 2, 2, 0.001, 0.125)
        self.assertEqual(size, 1)
        self.assertTrue(trials[0]["passed"])

    def test_minibatch_collates_compatible_crops_and_successors(self):
        with TemporaryDirectory() as tmp:
            movies = [Movie.open(make_movie(Path(tmp) / f"p{i}", 12, shape))
                      for i, shape in enumerate([(9, 11), (13, 15)])]
            clip, target = training_batch(movies, 4, 6, (20, 20), IntensityScale(0, 1),
                                          torch.device("cpu"), random.Random(4))
            self.assertEqual(clip.shape[:3], (6, 4, 1))
            self.assertEqual(clip[:, -1].shape, target.shape)
            torch.testing.assert_close(target[:, 0, 0, 0], clip[:, -1, 0, 0, 0] + 1)
            self.assertGreater(len(target[:, 0, 0, 0].unique()), 1)

    def test_final_decoder_learns_from_negative_features_and_loads_legacy_weights(self):
        model = FramePredictor(3, width=2)
        final_conv = model.up_fine[-2]
        with torch.no_grad():
            final_conv.weight.zero_()
            final_conv.bias.fill_(-1)
            model.change.weight.fill_(1)
        clip = torch.zeros(1, 1, 1, 3, 5, 7)
        prediction = model(clip)
        torch.testing.assert_close(prediction, torch.full_like(prediction, -0.02))
        prediction.mean().backward()
        torch.testing.assert_close(final_conv.bias.grad, torch.full_like(final_conv.bias, 0.01))

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save({"format": 1, "ndim": 3, "width": 2, "patch": [3, 5, 7],
                        "scale": {"offset": 0.0, "span": 1.0}, "model": model.state_dict()}, path)
            legacy, _, _ = load_checkpoint(path, torch.device("cpu"))
            torch.testing.assert_close(legacy(clip), clip[:, -1])

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
                          "--histories", "1", "2", "4", "--val-samples", "2", "--device", "cpu",
                          "--batch-size", "3" if len(shape) == 3 else "1"])
                self.assertEqual(json.loads((run / "run.json").read_text())["batch_size"], 3 if len(shape) == 3 else 1)
                self.assertTrue((run / "last.pt").is_file())
                checkpoint = str(run / "best.pt")
                loaded, _, _ = load_checkpoint(Path(checkpoint), torch.device("cpu"))
                self.assertIsInstance(loaded.up_fine[-1], torch.nn.LeakyReLU)
                self.assertEqual(loaded.up_fine[-1].negative_slope, 0.01)
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
