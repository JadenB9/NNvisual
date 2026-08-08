"""Reference MLP classifier 784-16-16-10 (relu, softmax) for the MLP page.

Exports public/data/weights/mlp-ref.bin (fp16) and a golden fixture: the
forward pass of fixture image 0 computed from the fp16-rounded weights, so
the JS engine can assert it reproduces the trainer bit-for-purpose."""

import json
from pathlib import Path

import numpy as np

from common import DenseNet, Adam, grad_check, write_nnw1, f16_round

root = Path(__file__).resolve().parent.parent


def load_nnd1(path):
    raw = path.read_bytes()
    assert raw[:4] == b"NND1"
    n = int.from_bytes(raw[4:8], "little")
    labels = np.frombuffer(raw, np.uint8, n, 8)
    pixels = np.frombuffer(raw, np.uint8, n * 784, 8 + n).reshape(n, 784)
    return pixels.astype(np.float64) / 255.0, labels.astype(np.int64)


X, Y = load_nnd1(root / "public/data/mnist-train.bin")
Xt, Yt = load_nnd1(root / "public/data/mnist-test.bin")

net = DenseNet([784, 32, 16, 10], task="classify", seed=7)

# sanity: analytic grads match numerical before we trust a long run
loss, dW, db = net.loss_and_grads(X[:8], Y[:8])
grad_check(lambda: net.loss(X[:8], Y[:8]), net.params(), dW + db)

opt = Adam(net.params(), lr=2e-3)
rng = np.random.default_rng(0)
batch = 64
for epoch in range(40):
    if epoch == 25:
        opt.lr = 6e-4
    order = rng.permutation(len(X))
    for s in range(0, len(X), batch):
        idx = order[s:s + batch]
        _, dW, db = net.loss_and_grads(X[idx], Y[idx])
        opt.step(net.params(), dW + db)
    if epoch % 5 == 4:
        print(f"epoch {epoch + 1}: test acc {net.accuracy(Xt, Yt):.4f}")

# evaluate what actually ships (fp16-rounded weights)
for i in range(len(net.W)):
    net.W[i] = f16_round(net.W[i]).astype(np.float64)
    net.b[i] = f16_round(net.b[i]).astype(np.float64)
acc = net.accuracy(Xt, Yt)
print(f"final fp16 test acc {acc:.4f}")
assert acc >= 0.93, "reference MLP under 93%, not shipping"

write_nnw1(
    root / "public/data/weights/mlp-ref.bin",
    {"kind": "mlp", "sizes": [784, 32, 16, 10], "hidden": "relu", "testAcc": round(acc, 4)},
    net.export_tensors(),
)

# golden fixture from the rounded weights
Xf, Yf = load_nnd1(root / "tests/fixtures/mnist-200.bin")
probs_all = net.forward(Xf)[-1]
m = probs_all - probs_all.max(axis=1, keepdims=True)
p = np.exp(m) / np.exp(m).sum(axis=1, keepdims=True)
fixture = {
    "image": Xf[0].tolist(),
    "label": int(Yf[0]),
    "probs": p[0].tolist(),
}
(root / "tests/fixtures/mlp-golden.json").write_text(json.dumps(fixture))
print("wrote tests/fixtures/mlp-golden.json")
