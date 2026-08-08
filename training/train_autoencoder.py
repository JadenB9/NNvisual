"""Autoencoder with a 2-D bottleneck for the latent-space demo.

encoder 784-256-64-2 (relu, relu, linear)
decoder 2-64-256-784 (relu, relu, linear)

The JS side loads these as two engine MLPs (task "regress"), whose hidden
layers are relu and whose output is linear — exactly this structure."""

import json
from pathlib import Path

import numpy as np

from common import Adam, grad_check, write_nnw1, f16_round

root = Path(__file__).resolve().parent.parent


def load_nnd1(path):
    raw = path.read_bytes()
    assert raw[:4] == b"NND1"
    n = int.from_bytes(raw[4:8], "little")
    labels = np.frombuffer(raw, np.uint8, n, 8)
    pixels = np.frombuffer(raw, np.uint8, n * 784, 8 + n).reshape(n, 784)
    return pixels.astype(np.float64) / 255.0, labels.astype(np.int64)


class Stack:
    """Affine stack, relu on all but the last layer (same as engine MLP)."""

    def __init__(self, sizes, rng):
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            self.W.append(rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i + 1], sizes[i])))
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, X):
        acts = [X]
        for i in range(len(self.W)):
            Z = acts[-1] @ self.W[i].T + self.b[i]
            acts.append(Z if i == len(self.W) - 1 else np.maximum(Z, 0))
        return acts

    def backward(self, acts, dOut):
        dW, db = [None] * len(self.W), [None] * len(self.W)
        delta = dOut
        for i in reversed(range(len(self.W))):
            dW[i] = delta.T @ acts[i]
            db[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i]) * (acts[i] > 0)
        dIn = dOut if len(self.W) == 0 else (delta @ self.W[0])
        return dW, db, dIn

    def params(self):
        return self.W + self.b


rng = np.random.default_rng(11)
enc = Stack([784, 256, 64, 2], rng)
dec = Stack([2, 64, 256, 784], rng)


def loss_and_grads(X):
    ea = enc.forward(X)
    da = dec.forward(ea[-1])
    diff = da[-1] - X
    loss = (diff ** 2).mean()
    dOut = 2 * diff / diff.size
    dWd, dbd, dz = dec.backward(da, dOut)
    dWe, dbe, _ = enc.backward(ea, dz)
    return loss, dWe + dbe + dWd + dbd


X, Y = load_nnd1(root / "public/data/mnist-train.bin")
Xt, Yt = load_nnd1(root / "public/data/mnist-test.bin")

loss, grads = loss_and_grads(X[:8])
grad_check(lambda: loss_and_grads(X[:8])[0], enc.params() + dec.params(), grads, samples=4, tol=1e-3)

params = enc.params() + dec.params()
opt = Adam(params, lr=1.2e-3)
order_rng = np.random.default_rng(0)
batch = 128
for epoch in range(80):
    if epoch == 50:
        opt.lr = 4e-4
    order = order_rng.permutation(len(X))
    for s in range(0, len(X), batch):
        _, grads = loss_and_grads(X[order[s:s + batch]])
        opt.step(params, grads)
    if epoch % 10 == 9:
        val = ((dec.forward(enc.forward(Xt)[-1])[-1] - Xt) ** 2).mean()
        print(f"epoch {epoch + 1}: val mse {val:.5f}")

# fp16-round what ships, then re-measure
for stack in (enc, dec):
    for i in range(len(stack.W)):
        stack.W[i] = f16_round(stack.W[i]).astype(np.float64)
        stack.b[i] = f16_round(stack.b[i]).astype(np.float64)
val = ((dec.forward(enc.forward(Xt)[-1])[-1] - Xt) ** 2).mean()
print(f"final fp16 val mse {val:.5f}")
# a 2-D bottleneck has a hard reconstruction floor around ~0.045 mse;
# the demo is about the latent plane, not pixel-perfect recon
assert val < 0.05, "autoencoder reconstruction too poor to ship"

tensors = []
for i in range(3):
    tensors.append((f"enc_w{i}", enc.W[i], "f16"))
    tensors.append((f"enc_b{i}", enc.b[i], "f16"))
    tensors.append((f"dec_w{i}", dec.W[i], "f16"))
    tensors.append((f"dec_b{i}", dec.b[i], "f16"))
write_nnw1(
    root / "public/data/weights/ae-ref.bin",
    {"kind": "autoencoder", "encSizes": [784, 256, 64, 2], "decSizes": [2, 64, 256, 784],
     "valMse": round(float(val), 5)},
    tensors,
)

raw = (root / "tests/fixtures/mnist-200.bin").read_bytes()
n = int.from_bytes(raw[4:8], "little")
pixels = np.frombuffer(raw, np.uint8, n * 784, 8 + n).reshape(n, 784) / 255.0
z = enc.forward(pixels[:1])[-1]
xhat = dec.forward(z)[-1]
fixture = {
    "latent": z[0].tolist(),
    "reconHead": xhat[0][:16].tolist(),
    "mse": float(((xhat[0] - pixels[0]) ** 2).mean()),
}
(root / "tests/fixtures/ae-golden.json").write_text(json.dumps(fixture))
print("wrote tests/fixtures/ae-golden.json")
