import numpy as np

from arc3 import perception as P


def test_components_and_centers():
    g = np.zeros((8, 8), dtype=np.int16)
    g[1:3, 1:3] = 5  # 2x2 block
    g[5, 5] = 7      # pixel
    g[6:8, 0:4] = 5  # 2x4 block (same color, separate)
    objs = P.components(g, ignore=(0,))
    assert [o.size for o in objs] == [4, 1, 8]
    assert objs[0].center == (1, 1) or objs[0].center == (2, 2) or objs[0].center == (1, 2)
    assert objs[0].shape_hash != objs[2].shape_hash  # different shapes
    assert objs[1].color == 7 and objs[1].center == (5, 5)
    g2 = np.roll(g, 2, axis=1)
    objs2 = P.components(g2, ignore=(0,))
    assert objs2[0].shape_hash == objs[0].shape_hash  # position-independent


def test_diff_and_moved():
    a = np.zeros((6, 6), dtype=np.int16)
    a[1, 1] = 3
    b = a.copy()
    b[1, 1] = 0
    b[1, 2] = 3
    d = P.diff(a, b)
    assert d.changed == 2 and d.bbox == (1, 1, 2, 1)
    assert d.transitions == {(3, 0): 1, (0, 3): 1}
    moved = P.moved_objects(P.components(a, ignore=(0,)), P.components(b, ignore=(0,)))
    assert len(moved) == 1 and (moved[0][2], moved[0][3]) == (1, 0)
    assert P.diff(a, a).empty


def test_scale_and_ascii_and_hash():
    small = np.arange(16, dtype=np.int16).reshape(4, 4)
    big = np.kron(small, np.ones((4, 4), dtype=np.int16))
    assert P.detect_scale(big) == 4
    ds, s = P.downscale(big)
    assert s == 4 and (ds == small).all()
    assert P.ascii(big).splitlines()[0] == "0123"
    assert P.grid_hash(big) == P.grid_hash(big.copy()) and P.grid_hash(big) != P.grid_hash(small)


def test_to_grid_accepts_frame_shapes():
    layer = [[1] * 64 for _ in range(64)]
    assert P.to_grid([layer, layer]).shape == (64, 64)
    assert P.to_grid(layer).shape == (64, 64)
    assert P.to_grid([np.zeros((64, 64))]).shape == (64, 64)


def test_render_png():
    g = np.zeros((4, 4), dtype=np.int16)
    png = P.render_png(g, scale=2)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_tile_map_is_one_char_per_tile():
    from arc3.perception import tile_map

    g = np.zeros((64, 64), dtype=np.int16)
    g[0:4, 0:4] = 9
    g[4:8, 4:8] = 3
    t = tile_map(g, 4)
    rows = t.split("\n")
    assert len(rows) == 16 and all(len(r) == 16 for r in rows)
    assert rows[0][0] == "9" and rows[1][1] == "3" and rows[0][1] == "0"
