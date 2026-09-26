import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.api.main import app
from backend.core.depth import backbone


def test_infer_relative_normalizes_shape_and_range(monkeypatch):
    class DummyModel:
        def eval(self):
            return self

    def fake_load_model(variant: str):
        return DummyModel()

    def fake_predict(model, image):
        h, w = image.shape[:2]
        values = np.linspace(0.1, 0.9, h * w, dtype=np.float32).reshape(h, w)
        return values

    monkeypatch.setattr(backbone, "load_model", fake_load_model)
    monkeypatch.setattr(backbone, "_predict_depth_map", fake_predict)

    image = np.zeros((32, 64, 3), dtype=np.uint8)
    depth = backbone.infer_relative(image)

    assert depth.dtype == np.float32
    assert depth.shape == (32, 64)
    assert depth.min() >= 0.0
    assert depth.max() <= 1.0
    assert np.all(np.isfinite(depth))


def test_depth_preview_endpoint_returns_png(monkeypatch):
    def fake_infer_relative(image):
        return np.linspace(0.0, 1.0, 32 * 48, dtype=np.float32).reshape(32, 48)

    monkeypatch.setattr("backend.api.main.infer_relative", fake_infer_relative)

    payload = np.zeros((32, 48, 3), dtype=np.uint8)
    png_bytes = io.BytesIO()
    Image.fromarray(payload, mode="RGB").save(png_bytes, format="PNG")

    client = TestClient(app)
    response = client.post(
        "/v1/depth/preview",
        files={"file": ("sample.png", png_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    rendered = Image.open(io.BytesIO(response.content))
    assert rendered.size == (48, 32)
