"""Shared pieces for the offline trainers: file writers matching the JS
loaders, a dense net with the exact same conventions as js/engine/mlp.js,
Adam, and a numerical gradient checker every trainer runs before training."""

import gzip
import json
import struct

import numpy as np


# ---------- file formats (mirrors js/engine/weights.js and mnist.js) ----------

def write_nnw1(path, meta, tensors):
    """tensors: list of (name, ndarray, dtype) with dtype f16|f32|u8."""
    payload = b""
    specs = []
    for name, arr, dt in tensors:
        if dt == "f16":
            data = arr.astype(np.float16).tobytes()
        elif dt == "f32":
            data = arr.astype(np.float32).tobytes()
        elif dt == "u8":
            data = arr.astype(np.uint8).tobytes()
        else:
            raise ValueError(dt)
        specs.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": dt,
            "offset": len(payload),
            "count": int(arr.size),
        })
        payload += data
        payload += b"\0" * ((-len(payload)) % 4)
    header = json.dumps({"meta": meta, "tensors": specs}).encode()
    with open(path, "wb") as f:
        f.write(b"NNW1")
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        f.write(payload)
    print(f"wrote {path}")


def write_nnd1(path, images, labels):
    """images: (N, 784) uint8; labels: (N,) uint8."""
    n = len(labels)
    with open(path, "wb") as f:
        f.write(b"NND1")
        f.write(struct.pack("<I", n))
        f.write(labels.astype(np.uint8).tobytes())
        f.write(images.astype(np.uint8).tobytes())
    print(f"wrote {path} ({n} images)")


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)


def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8)


def f16_round(a):
    """Round to fp16 and back — goldens are computed from what actually ships."""
    return a.astype(np.float16).astype(np.float32)


# ---------- dense net (same conventions as js/engine/mlp.js) ----------

class DenseNet:
    """W[l]: (nOut, nIn) row-per-output-unit; relu hidden; softmax or linear head."""

    def __init__(self, sizes, task="classify", seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.task = task
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])
            self.W.append(rng.normal(0, scale, (sizes[i + 1], sizes[i])).astype(np.float64))
            self.b.append(np.zeros(sizes[i + 1]))

    def params(self):
        return self.W + self.b

    def forward(self, X):
        acts = [X]
        for i in range(len(self.W)):
            Z = acts[-1] @ self.W[i].T + self.b[i]
            acts.append(Z if i == len(self.W) - 1 else np.maximum(Z, 0))
        return acts

    def loss_and_grads(self, X, Y):
        """Y: int labels (classify) or float targets (regress). Returns
        (loss, dW list, db list) for the mean batch loss."""
        n = len(X)
        acts = self.forward(X)
        out = acts[-1]
        if self.task == "classify":
            m = out - out.max(axis=1, keepdims=True)
            e = np.exp(m)
            p = e / e.sum(axis=1, keepdims=True)
            loss = -np.log(np.clip(p[np.arange(n), Y], 1e-12, None)).mean()
            delta = p.copy()
            delta[np.arange(n), Y] -= 1
            delta /= n
        else:
            diff = out - Y
            loss = (diff ** 2).mean()
            delta = 2 * diff / diff.size

        dW = [None] * len(self.W)
        db = [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            dW[i] = delta.T @ acts[i]
            db[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i]) * (acts[i] > 0)
        return loss, dW, db

    def loss(self, X, Y):
        return self.loss_and_grads(X, Y)[0]

    def accuracy(self, X, Y):
        return float((self.forward(X)[-1].argmax(axis=1) == Y).mean())

    def export_tensors(self, prefix=""):
        out = []
        for i in range(len(self.W)):
            out.append((f"{prefix}w{i}", self.W[i], "f16"))
            out.append((f"{prefix}b{i}", self.b[i], "f16"))
        return out


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def grad_check(loss_fn, params, grads, samples=6, h=1e-5, tol=1e-4, rng=None):
    """Spot-check a few entries of every param tensor against central
    differences. loss_fn() recomputes the loss from current param values."""
    rng = rng or np.random.default_rng(0)
    for pi, (p, g) in enumerate(zip(params, grads)):
        flat_p = p.reshape(-1)
        flat_g = g.reshape(-1)
        for _ in range(min(samples, flat_p.size)):
            i = rng.integers(flat_p.size)
            saved = flat_p[i]
            flat_p[i] = saved + h
            up = loss_fn()
            flat_p[i] = saved - h
            down = loss_fn()
            flat_p[i] = saved
            numeric = (up - down) / (2 * h)
            denom = max(abs(numeric), abs(flat_g[i]), 1e-8)
            rel = abs(numeric - flat_g[i]) / denom
            assert rel < tol or abs(numeric - flat_g[i]) < 1e-7, (
                f"grad check failed: param {pi} idx {i}: numeric {numeric} vs analytic {flat_g[i]}")
    print("grad check ok")


def read_nnw1(path):
    """Read an NNW1 weight file back into {name: ndarray} (f32)."""
    raw = open(path, "rb").read()
    assert raw[:4] == b"NNW1"
    hlen = struct.unpack("<I", raw[4:8])[0]
    header = json.loads(raw[8:8 + hlen])
    payload = raw[8 + hlen:]
    out = {}
    for t in header["tensors"]:
        off = t["offset"]
        if t["dtype"] == "f16":
            a = np.frombuffer(payload, np.float16, t["count"], off).astype(np.float64)
        elif t["dtype"] == "f32":
            a = np.frombuffer(payload, np.float32, t["count"], off).astype(np.float64)
        else:
            a = np.frombuffer(payload, np.uint8, t["count"], off).astype(np.float64)
        out[t["name"]] = a.reshape(t["shape"])
    return header["meta"], out
