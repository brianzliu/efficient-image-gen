import json
import tempfile
import unittest
from pathlib import Path

from evaluation.experiment import BaselineCache, baseline_fields, load_completed, sha256_file, trial_key

IDENTITY = dict(
    experiment="curvature-quick", prompts=["a fox", "a mug"], seeds=[42], repeats=1, configs=["full-50", "taylor-3"],
    size=512, guidance=3.5, scheduler="flow_match_euler_discrete", source_sha256="abc",
    checkpoint="ckpt", checkpoint_revision="rev", packages={"mlx": "0.32.4", "mflux": "0.24.0", "numpy": "2.5.4"},
    macos="26.5.1", machine="arm64", hardware="Apple M4",
)


def fields(identity=IDENTITY, **overrides):
    trial = dict(config="full-50", steps=50, prompt="a fox", seed=42, repeat=0) | overrides
    return baseline_fields(identity, trial["config"], trial["steps"], trial["prompt"], trial["seed"], trial["repeat"])


class Directory(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def image(self, name: str, content: bytes = b"png bytes") -> Path:
        path = self.root / "images" / name
        path.write_bytes(content)
        return path


class BaselineCacheTest(Directory):
    def setUp(self) -> None:
        super().setUp()
        self.cache = BaselineCache(self.root / "cache")
        self.row = dict(key="r0-p0-s42-full-50", total_s=600.0, cached_from=None)
        self.cache.store(fields(), self.row, self.image("baseline.png"))

    def test_hit_returns_stored_measurement_and_verified_image(self) -> None:
        key, row, image = self.cache.lookup(fields())
        self.assertEqual(key, BaselineCache.key(fields()))
        self.assertEqual(row, self.row)
        self.assertEqual(image.read_bytes(), b"png bytes")

    def test_any_changed_generation_setting_misses(self) -> None:
        changes = dict(source_sha256="def", size=1024, guidance=4.0, scheduler="linear", checkpoint_revision="other",
                       packages={**IDENTITY["packages"], "mlx": "0.33.0"}, macos="27.0", hardware="Apple M1 Pro")
        for field, value in changes.items():
            with self.subTest(field=field):
                self.assertIsNone(self.cache.lookup(fields(IDENTITY | {field: value})))
        for field, value in dict(prompt="a mug", seed=43, repeat=1, steps=17, config="full-17").items():
            with self.subTest(field=field):
                self.assertIsNone(self.cache.lookup(fields(**{field: value})))

    def test_experiment_level_choices_do_not_affect_reuse(self) -> None:
        other = IDENTITY | dict(experiment="curvature-validate", prompts=["x", "a fox"], configs=["full-50"], seeds=[7, 42])
        self.assertIsNotNone(self.cache.lookup(fields(other)))

    def test_missing_identity_field_is_an_error(self) -> None:
        identity = {k: v for k, v in IDENTITY.items() if k != "source_sha256"}
        with self.assertRaises(KeyError):
            fields(identity)

    def test_modified_cached_image_is_refused(self) -> None:
        (self.root / "cache" / BaselineCache.key(fields()) / "image.png").write_bytes(b"edited")
        with self.assertRaisesRegex(ValueError, "modified"):
            self.cache.lookup(fields())

    def test_entry_whose_fields_disagree_with_its_key_is_refused(self) -> None:
        entry = self.root / "cache" / BaselineCache.key(fields()) / "entry.json"
        record = json.loads(entry.read_text())
        record["fields"]["seed"] = 43
        entry.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.cache.lookup(fields())

    def test_store_keeps_existing_entry_and_leaves_no_staging(self) -> None:
        self.cache.store(fields(), dict(self.row, total_s=1.0), self.image("other.png", b"different"))
        self.assertEqual(self.cache.lookup(fields())[1]["total_s"], 600.0)
        self.assertEqual([p.name for p in (self.root / "cache").iterdir()], [BaselineCache.key(fields())])


class ResumeTest(Directory):
    def row(self, config="full-50", prompt_id=0, **overrides) -> dict:
        key = trial_key(0, prompt_id, 42, config)
        image = self.image(f"{key}.png", key.encode())
        return dict(key=key, repeat=0, prompt_id=prompt_id, prompt=IDENTITY["prompts"][prompt_id], seed=42,
                    config=config, image=f"images/{key}.png", image_sha256=sha256_file(image)) | overrides

    def write(self, *rows, tail: str = "") -> Path:
        path = self.root / "measurements.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows) + tail)
        return path

    def test_no_measurements_means_nothing_done(self) -> None:
        self.assertEqual(load_completed(self.root / "measurements.jsonl", self.root, IDENTITY), {})

    def test_valid_rows_are_skipped_on_resume(self) -> None:
        rows = [self.row(), self.row("taylor-3"), self.row(prompt_id=1)]
        done = load_completed(self.write(*rows), self.root, IDENTITY)
        self.assertEqual(set(done), {r["key"] for r in rows})

    def test_interrupted_final_line_is_dropped_so_that_trial_reruns(self) -> None:
        path = self.write(self.row(), tail='{"key": "r0-p0-s42-taylor-3", "tot')
        done = load_completed(path, self.root, IDENTITY)
        self.assertEqual(set(done), {"r0-p0-s42-full-50"})
        self.assertTrue(path.read_text().endswith("\n"))
        self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_missing_or_changed_image_is_an_error(self) -> None:
        row = self.row()
        path = self.write(row)
        (self.root / row["image"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_completed(path, self.root, IDENTITY)
        (self.root / row["image"]).unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            load_completed(path, self.root, IDENTITY)

    def test_rows_without_a_recorded_hash_are_not_trusted(self) -> None:
        row = self.row()
        del row["image_sha256"]
        with self.assertRaises(ValueError):
            load_completed(self.write(row), self.root, IDENTITY)

    def test_duplicate_or_foreign_rows_are_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_completed(self.write(self.row(), self.row()), self.root, IDENTITY)
        for bad in [self.row(prompt="another prompt"), self.row("taylor2-3-c0.5"), self.row(seed=43),
                    self.row(key="r0-p1-s42-full-50")]:
            with self.subTest(row=bad), self.assertRaisesRegex(ValueError, "identity"):
                load_completed(self.write(bad), self.root, IDENTITY)


if __name__ == "__main__":
    unittest.main()
