from collections import OrderedDict

import numpy as np
from PIL import Image
import torch

from dinotool.agent_tool import _dense_service_parser, request_to_argv
from dinotool.fast_dense import FastDenseConfig, FastDenseSession, _normalize_classes
from dinotool.prompts import ClassSpec


def test_fast_dense_config_validates_latency_resolution() -> None:
    FastDenseConfig(input_resolution=512).validate()
    try:
        FastDenseConfig(input_resolution=510).validate()
    except ValueError as error:
        assert "patch size" in str(error)
    else:
        raise AssertionError("Expected non-divisible resolution to be rejected")


def test_fast_dense_normalizes_background_without_reordering_foreground() -> None:
    classes, added = _normalize_classes((ClassSpec("road", ("road",)), ClassSpec("building", ("building",))))
    assert added is True
    assert [spec.name for spec in classes] == ["background", "road", "building"]


def test_dense_logits_are_upsampled_and_thresholded() -> None:
    logits = torch.tensor([[[[4.0]], [[0.0]]]])
    session = object.__new__(FastDenseSession)
    session.config = FastDenseConfig(output_temperature=1.0, confidence_threshold=0.99)
    session.model = type("FakeModel", (), {"device": torch.device("cpu"), "use_amp": False})()
    session._text_cache = {}
    session._class_text_cache = {}
    session._image_cache = OrderedDict()

    # Use the pure model-independent operation through a tiny fake model so
    # the test does not need foundation weights.
    class FakeModel:
        device = torch.device("cpu")
        use_amp = False

        def encode_image(self, rgb):
            return torch.ones(1, 2, 1, 1), torch.ones(1, 2)

        def encode_text(self, classes):
            return torch.eye(2)[: len(classes)]

        def similarity_logits(self, features, text):
            return logits

    session.model = FakeModel()
    image = Image.new("RGB", (4, 6), color=(80, 90, 100))
    labels, confidence, metadata = session.segment(
        image,
        (ClassSpec("background", ("background",)), ClassSpec("road", ("road",))),
    )
    assert labels.shape == (6, 4)
    assert confidence.shape == (6, 4)
    assert np.all(labels == 255)
    assert metadata["patch_grid"] == [1, 1]


def test_fast_dense_image_and_text_caches_reuse_entries() -> None:
    class FakeModel:
        device = torch.device("cpu")
        use_amp = False
        calls = 0

        def encode_image(self, rgb):
            self.calls += 1
            return torch.ones(1, 2, 1, 1), torch.ones(1, 2)

        def encode_text(self, classes):
            return torch.ones(len(classes), 2)

        def similarity_logits(self, features, text):
            return torch.zeros(1, text.shape[0], 1, 1)

    session = object.__new__(FastDenseSession)
    session.config = FastDenseConfig(input_resolution=16, image_cache_size=1)
    session.model = FakeModel()
    session._text_cache = {}
    session._class_text_cache = {}
    session._image_cache = OrderedDict()
    image = Image.new("RGB", (16, 16), color=(1, 2, 3))
    classes = (ClassSpec("background", ("background",)), ClassSpec("road", ("road",)))

    session.segment(image, classes, image_key="same")
    _, image_hit, _ = session._image_entry(image, "same")
    _, text_hit, class_hits = session._text_features(classes)
    assert image_hit is True
    assert text_hit is True
    assert class_hits == len(classes)
    assert session.model.calls == 1


def test_fast_dense_agent_cli_and_json_mode() -> None:
    args = _dense_service_parser().parse_args(["--serve-dino-dense", "--warmup"])
    assert args.warmup is True
    argv = request_to_argv(
        {
            "mode": "fast-dense",
            "image": "scene.png",
            "classes": ["road", "building"],
            "output_dir": "out",
            "dino_input_resolution": 512,
        }
    )
    assert argv[:1] == ["infer-dino-dense"]
    assert "--dino-input-resolution" in argv
