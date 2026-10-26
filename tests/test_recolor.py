import cv2
import numpy as np

from pipette_coach.recolor import dye_weight, recolor


def _hsv_px(h, s, v):
    return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]


def _scene():
    """4 x 4 patches: red dye (hue 175), red dye across the wrap (hue 1), skin (hue 8), pale grey plastic."""
    img = np.zeros((40, 40, 3), np.uint8)
    img[:20, :20] = _hsv_px(175, 180, 160)
    img[:20, 20:] = _hsv_px(1, 170, 150)
    img[20:, :20] = _hsv_px(8, 140, 150)
    img[20:, 20:] = _hsv_px(100, 15, 200)
    return img


def test_dye_weight_finds_red_dye_not_skin_or_plastic():
    w = dye_weight(_scene())
    assert w[5:15, 5:15].min() > 0.95 and w[5:15, 25:35].min() > 0.95
    assert w[25:35, 5:15].max() < 0.01 and w[25:35, 25:35].max() < 0.01


def test_pink_skin_with_the_dye_hue_is_not_dye():
    lab = np.zeros((10, 10, 3), np.uint8)
    lab[:] = (160, 128 + 29, 128 + 15)  # warm, dull pink: Lab angle ~27 deg, chroma ~33
    skin = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    h = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)[5, 5, 0]
    assert h >= 170 or h <= 2  # same HSV hue as the dye ...
    assert dye_weight(skin).max() < 0.01  # ... but not dye


def test_recolor_changes_only_the_dye():
    img = _scene()
    out = recolor(img, dye_weight(img), "blue")
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    assert abs(int(hsv[10, 10, 0]) - 112) <= 2
    assert np.array_equal(out[25:35, :], img[25:35, :])
    clear = cv2.cvtColor(recolor(img, dye_weight(img), "clear"), cv2.COLOR_BGR2HSV)
    assert clear[10, 10, 1] < 25
