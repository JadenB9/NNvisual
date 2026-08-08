"""Reference convnet for the CNN page:
conv(1->8,3x3) relu pool conv(8->16,3x3) relu pool fc(400->10) softmax.

Exports public/data/weights/cnn-ref.bin (fp16) and a golden fixture with
the probabilities AND the flattened pool2 stage, pinning the JS stage
implementations (conv, pool, flatten order) to the trainer."""

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
    pixels = np.frombuffer(raw, np.uint8, n * 784, 8 + n).reshape(n, 1, 28, 28)
    return pixels.astype(np.float64) / 255.0, labels.astype(np.int64)


def conv_fwd(X, W, b):
    B, C, H, Wd = X.shape
    OC, _, KH, KW = W.shape
    OH, OW = H - KH + 1, Wd - KW + 1
    Z = np.zeros((B, OC, OH, OW))
    for ky in range(KH):
        for kx in range(KW):
            Z += np.einsum("bchw,oc->bohw", X[:, :, ky:ky + OH, kx:kx + OW], W[:, :, ky, kx])
    return Z + b[None, :, None, None]


def conv_bwd(X, W, dZ):
    B, C, H, Wd = X.shape
    OC, _, KH, KW = W.shape
    OH, OW = dZ.shape[2], dZ.shape[3]
    dW = np.zeros_like(W)
    dX = np.zeros_like(X)
    for ky in range(KH):
        for kx in range(KW):
            dW[:, :, ky, kx] = np.einsum("bohw,bchw->oc", dZ, X[:, :, ky:ky + OH, kx:kx + OW])
            dX[:, :, ky:ky + OH, kx:kx + OW] += np.einsum("bohw,oc->bchw", dZ, W[:, :, ky, kx])
    db = dZ.sum(axis=(0, 2, 3))
    return dW, db, dX


def pool_fwd(X):
    """First-max-wins pooling, matching js/engine/conv.js tie-breaking:
    windows flatten row-major (dy, dx) and argmax picks the first max."""
    B, C, H, W = X.shape
    OH, OW = H // 2, W // 2
    Xc = (X[:, :, :OH * 2, :OW * 2]
          .reshape(B, C, OH, 2, OW, 2)
          .transpose(0, 1, 2, 4, 3, 5)
          .reshape(B, C, OH, OW, 4))
    idx = Xc.argmax(-1)
    out = np.take_along_axis(Xc, idx[..., None], -1)[..., 0]
    onehot = np.zeros_like(Xc)
    np.put_along_axis(onehot, idx[..., None], 1.0, -1)
    return out, onehot


def pool_bwd(dOut, onehot, shape):
    B, C, H, W = shape
    OH, OW = H // 2, W // 2
    dXc = onehot * dOut[..., None]
    dX = np.zeros(shape)
    dX[:, :, :OH * 2, :OW * 2] = (dXc
        .reshape(B, C, OH, OW, 2, 2)
        .transpose(0, 1, 2, 4, 3, 5)
        .reshape(B, C, OH * 2, OW * 2))
    return dX


class ConvNet:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / 9), (8, 1, 3, 3))
        self.b1 = np.zeros(8)
        self.W2 = rng.normal(0, np.sqrt(2 / (8 * 9)), (16, 8, 3, 3))
        self.b2 = np.zeros(16)
        self.Wf = rng.normal(0, np.sqrt(2 / 400), (10, 400))
        self.bf = np.zeros(10)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.Wf, self.bf]

    def forward(self, X):
        c = {}
        c["z1"] = conv_fwd(X, self.W1, self.b1)          # B,8,26,26
        c["a1"] = np.maximum(c["z1"], 0)
        c["p1"], c["m1"] = pool_fwd(c["a1"])             # B,8,13,13
        c["z2"] = conv_fwd(c["p1"], self.W2, self.b2)    # B,16,11,11
        c["a2"] = np.maximum(c["z2"], 0)
        c["p2"], c["m2"] = pool_fwd(c["a2"])             # B,16,5,5
        c["flat"] = c["p2"].reshape(len(X), -1)          # B,400
        c["logits"] = c["flat"] @ self.Wf.T + self.bf
        return c

    def loss_and_grads(self, X, Y):
        n = len(X)
        c = self.forward(X)
        m = c["logits"] - c["logits"].max(axis=1, keepdims=True)
        e = np.exp(m)
        p = e / e.sum(axis=1, keepdims=True)
        loss = -np.log(np.clip(p[np.arange(n), Y], 1e-12, None)).mean()
        delta = p.copy()
        delta[np.arange(n), Y] -= 1
        delta /= n

        dWf = delta.T @ c["flat"]
        dbf = delta.sum(axis=0)
        dflat = delta @ self.Wf
        dp2 = dflat.reshape(c["p2"].shape)
        da2 = pool_bwd(dp2, c["m2"], c["a2"].shape) * (c["a2"] > 0)
        dW2, db2, dp1 = conv_bwd(c["p1"], self.W2, da2)
        da1 = pool_bwd(dp1, c["m1"], c["a1"].shape) * (c["a1"] > 0)
        dW1, db1, _ = conv_bwd(X, self.W1, da1)
        return loss, [dW1, db1, dW2, db2, dWf, dbf]

    def accuracy(self, X, Y, batch=500):
        hits = 0
        for s in range(0, len(X), batch):
            c = self.forward(X[s:s + batch])
            hits += int((c["logits"].argmax(axis=1) == Y[s:s + batch]).sum())
        return hits / len(X)


X, Y = load_nnd1(root / "public/data/mnist-train.bin")
Xt, Yt = load_nnd1(root / "public/data/mnist-test.bin")

net = ConvNet(seed=3)
# grad-check on tie-free random input: relu'd zeros and pooling ties make
# numerical checks on real images ill-defined at kink points
gc_rng = np.random.default_rng(9)
Xg = gc_rng.uniform(0.05, 1.0, (2, 1, 28, 28))
Yg = gc_rng.integers(0, 10, 2)
loss, grads = net.loss_and_grads(Xg, Yg)
grad_check(lambda: net.loss_and_grads(Xg, Yg)[0], net.params(), grads, samples=4, tol=1e-3)

opt = Adam(net.params(), lr=1.5e-3)
rng = np.random.default_rng(0)
batch = 64
for epoch in range(16):
    if epoch == 10:
        opt.lr = 4e-4
    order = rng.permutation(len(X))
    for s in range(0, len(X), batch):
        idx = order[s:s + batch]
        _, grads = net.loss_and_grads(X[idx], Y[idx])
        opt.step(net.params(), grads)
    print(f"epoch {epoch + 1}: test acc {net.accuracy(Xt, Yt):.4f}")

for name in ["W1", "b1", "W2", "b2", "Wf", "bf"]:
    setattr(net, name, f16_round(getattr(net, name)).astype(np.float64))
acc = net.accuracy(Xt, Yt)
print(f"final fp16 test acc {acc:.4f}")
assert acc >= 0.95, "reference CNN under 95%, not shipping"

write_nnw1(
    root / "public/data/weights/cnn-ref.bin",
    {"kind": "cnn", "testAcc": round(acc, 4)},
    [
        ("c1w", net.W1, "f16"), ("c1b", net.b1, "f16"),
        ("c2w", net.W2, "f16"), ("c2b", net.b2, "f16"),
        ("fcw", net.Wf, "f16"), ("fcb", net.bf, "f16"),
    ],
)

def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()

raw = (root / "tests/fixtures/mnist-200.bin").read_bytes()
n = int.from_bytes(raw[4:8], "little")
labels = np.frombuffer(raw, np.uint8, n, 8)
pixels = np.frombuffer(raw, np.uint8, n * 784, 8 + n).reshape(n, 1, 28, 28) / 255.0

c = net.forward(pixels[:1])
fixture = {
    "label": int(labels[0]),
    "probs": softmax(c["logits"][0]).tolist(),
    "flatHead": c["flat"][0][:16].tolist(),
}
(root / "tests/fixtures/cnn-golden.json").write_text(json.dumps(fixture))
print("wrote tests/fixtures/cnn-golden.json")
