"""Render the Well Plate UI's Web Audio sounds (wellPlateUiWeb/static/app.js: note/bell/goSound/warnSound/
dispensedSound) to WAV files for the demo video, note for note. Run: python make_sfx.py"""

import wave
from pathlib import Path

import numpy as np

SR = 48000


def note(buf, f, at, length, type="sine", gain=0.15, attack=0.005, lowpass=None, detune=0.0):
    n0, n = int(at * SR), int((length + 0.05) * SR)
    t = np.arange(n) / SR
    if isinstance(f, (list, tuple)):  # exponential glide f0 -> f1 over length, then hold
        tt = np.minimum(t, length)
        freq = f[0] * (f[1] / f[0]) ** (tt / length)
    else:
        freq = np.full(n, float(f))
    freq = freq * 2 ** (detune / 1200)
    phase = np.cumsum(freq) / SR
    if type == "sine":
        x = np.sin(2 * np.pi * phase)
    elif type == "sawtooth":
        x = 2 * (phase % 1) - 1
    elif type == "square":
        x = np.sign(np.sin(2 * np.pi * phase))
    else:  # triangle
        x = 2 * np.abs(2 * (phase % 1) - 1) - 1
    # gain: 0.0001 -> gain (exponential, attack), then -> 0.0001 at length (exponential)
    rise = 0.0001 * (gain / 0.0001) ** (np.minimum(t, attack) / attack)
    g = np.where(t < attack, rise, gain * (0.0001 / gain) ** (np.clip(t - attack, 0, None) / max(length - attack, 1e-6)))
    g[t > length] = 0.0001
    y = x * g
    if lowpass:  # RBJ biquad lowpass, Q = 1/sqrt(2) (Web Audio's default)
        w0 = 2 * np.pi * lowpass / SR
        alpha = np.sin(w0) / (2 * 0.7071)
        b = np.array([(1 - np.cos(w0)) / 2, 1 - np.cos(w0), (1 - np.cos(w0)) / 2])
        a = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])
        b, a = b / a[0], a / a[0]
        out = np.zeros_like(y)
        x1 = x2 = y1 = y2 = 0.0
        for i, v in enumerate(y):
            o = b[0] * v + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            x2, x1, y2, y1 = x1, v, y1, o
            out[i] = o
        y = out
    end = min(len(buf), n0 + n)
    buf[n0:end] += y[: end - n0]


def bell(buf, f, at, length, gain):
    note(buf, f, at, length, gain=gain)
    note(buf, f * 2.76, at, length * 0.5, gain=gain * 0.22)
    note(buf, f * 5.4, at, length * 0.25, gain=gain * 0.07)


def go(buf):
    bell(buf, 1046.5, 0, 0.35, 0.16)
    bell(buf, 1568, 0.09, 0.5, 0.16)


def warn(buf):
    for at in (0, 0.2):
        note(buf, [233, 196], at, 0.17, type="sawtooth", gain=0.13, lowpass=1500)
        note(buf, [247, 207], at, 0.17, type="sawtooth", gain=0.1, lowpass=1500)


def dispensed(buf):
    note(buf, [190, 60], 0, 0.14, gain=0.22)
    for k, f in enumerate([1046.5, 1318.5, 1568, 2093]):
        bell(buf, f, 0.03 + k * 0.06, 0.45, 0.11)


def save(name, fn, seconds):
    buf = np.zeros(int(seconds * SR))
    fn(buf)
    buf = np.tanh(buf * 1.6) / np.tanh(1.6)  # stands in for the app's DynamicsCompressor: louder, no clipping
    pcm = (np.clip(buf / max(np.abs(buf).max(), 1e-6) * 0.9, -1, 1) * 32767).astype(np.int16)
    path = Path(__file__).parent / "public" / "sfx" / f"{name}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(path)


if __name__ == "__main__":
    save("go", go, 0.7)
    save("warn", warn, 0.5)
    save("dispensed", dispensed, 0.8)
