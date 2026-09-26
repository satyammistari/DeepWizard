import numpy as np

from backend.core.depth.tile_infer import infer_and_stitch


def test_infer_and_stitch_synthetic_image():
    h, w = 300, 400
    # synthetic RGB gradient
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    img = np.stack([xx, yy, (xx + yy) // 2], axis=-1)

    stitched = infer_and_stitch(img, tile_size=128, overlap_frac=0.25)
    assert stitched.shape == (h, w)
    assert np.isfinite(stitched).any()
    # at least 50% pixels finite
    finite_frac = float(np.isfinite(stitched).mean())
    assert finite_frac > 0.5
