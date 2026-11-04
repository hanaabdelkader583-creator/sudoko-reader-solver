import numpy as np
import config
from src.data.mosaic import load_raw_test, render_mosaic, split_tiles
from src.nn.interface import predict_cells


def test_predict_cells_on_a_real_mosaic():
    test_images, _ = load_raw_test()
    d = np.load(config.SUDOKU_DIR / "puzzles_g35.npz")
    P, I = d["puzzles"][0], d["cell_img_idx"][0]
    probs, blank = predict_cells(split_tiles(render_mosaic(I, test_images)))

    assert probs.shape == (81, 9) and blank.shape == (81,)
    assert (blank == (P == 0)).all()
    given = ~blank
    assert np.allclose(probs[given].sum(axis=1), 1.0, atol=1e-5)
    assert (probs[blank] == 0).all()

    read = probs[given].argmax(axis=1) + 1
    acc = (read == P[given]).mean()
    print("cell accuracy on this mosaic:", acc)
    assert acc >= 0.9