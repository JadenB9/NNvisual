"""Class-conditional diffusion for the diffusion page — x0-parameterization.

The denoiser predicts the CLEAN IMAGE x0-hat from (x_t, t, class), not the
noise. An MLP narrower than 784 cannot express the near-identity map that
eps-prediction needs at high t (rank floor ~= 1 - width/784, which is
exactly the 0.257 plateau the eps run hit), but a digit image is highly
compressible, so x0-prediction fits easily. The sampler derives
  eps-hat = (x_t - sqrt(abar)*x0-hat) / sqrt(1-abar)
and DDIM proceeds unchanged.

Model: MLP [784+64+64, 512, 512, 512, 784], tanh hidden, linear out.
Class index 10 is the null class for classifier-free guidance (10% dropout).
Schedule and t-embedding mirror js/engine/diffusion.js exactly."""

import json
from pathlib import Path

import numpy as np

from common import Adam, grad_check, write_nnw1, f16_round, read_nnw1, \
    read_idx_images, read_idx_labels

root = Path(__file__).resolve().parent.parent
T = 200
TDIM = 64
CDIM = 64
NULL_CLASS = 10
SIZES = [784 + TDIM + CDIM, 512, 512, 512, 784]


def make_schedule(steps=T, s=0.008):
    t = np.arange(steps + 1)
    f = np.cos(((t / steps + s) / (1 + s)) * np.pi / 2) ** 2
    return np.clip(f / f[0], 1e-5, 1.0)


def t_embed(ts):
    half = TDIM // 2
    f = 10000.0 ** (-np.arange(half) / half)
    ang = np.asarray(ts, dtype=np.float64)[:, None] * f[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


class Stack:
    """tanh hidden, linear out — mirrors engine MLP with hidden:'tanh'."""

    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(1 / sizes[i]), (sizes[i + 1], sizes[i]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def forward(self, X):
        acts = [X]
        for i in range(len(self.W)):
            Z = acts[-1] @ self.W[i].T + self.b[i]
            acts.append(Z if i == len(self.W) - 1 else np.tanh(Z))
        return acts

    def backward(self, acts, dOut):
        dW, db = [None] * len(self.W), [None] * len(self.W)
        delta = dOut
        for i in reversed(range(len(self.W))):
            dW[i] = delta.T @ acts[i]
            db[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i]) * (1 - acts[i] ** 2)
        dIn = delta @ self.W[0]
        return dW, db, dIn

    def params(self):
        return self.W + self.b


rng = np.random.default_rng(21)
net = Stack(SIZES, rng)
CE = rng.normal(0, 0.5, (11, CDIM))
abar = make_schedule()


def batch_inputs(x0, cls, ts, eps):
    a = abar[ts][:, None]
    xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps
    return np.concatenate([xt, t_embed(ts), CE[cls]], axis=1)


def loss_and_grads(x0, cls, ts, eps):
    inp = batch_inputs(x0, cls, ts, eps)
    acts = net.forward(inp)
    diff = acts[-1] - x0
    loss = (diff ** 2).mean()
    dOut = 2 * diff / diff.size
    dW, db, dIn = net.backward(acts, dOut)
    dCE = np.zeros_like(CE)
    np.add.at(dCE, cls, dIn[:, 784 + TDIM:])
    return loss, dW + db + [dCE]


x0c = rng.uniform(-1, 1, (4, 784))
clsc = np.array([1, 5, 10, 3])
tsc = np.array([10, 100, 150, 199])
epsc = rng.normal(0, 1, (4, 784))
loss, grads = loss_and_grads(x0c, clsc, tsc, epsc)
grad_check(lambda: loss_and_grads(x0c, clsc, tsc, epsc)[0],
           net.params() + [CE], grads, samples=4, tol=1e-3)

data_dir = Path(__file__).parent / "data"
images = read_idx_images(data_dir / "train-images-idx3-ubyte.gz").astype(np.float64) / 255.0 * 2 - 1
labels = read_idx_labels(data_dir / "train-labels-idx1-ubyte.gz").astype(np.int64)
test_images = read_idx_images(data_dir / "t10k-images-idx3-ubyte.gz").astype(np.float64) / 255.0 * 2 - 1
test_labels = read_idx_labels(data_dir / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
print(f"training x0-denoiser on {len(images)} images")


def binned_eval(n=768, seed=7):
    er = np.random.default_rng(seed)
    out = {}
    for name, lo, hi in [("low", 1, 40), ("mid", 40, 180), ("high", 180, 200)]:
        ts = er.integers(lo, hi + 1, n)
        eps = er.standard_normal((n, 784))
        inp = batch_inputs(test_images[:n], test_labels[:n], ts, eps)
        out[name] = float(((net.forward(inp)[-1] - test_images[:n]) ** 2).mean())
    return out


params = net.params() + [CE]
opt = Adam(params, lr=1e-3)
order_rng = np.random.default_rng(0)
noise_rng = np.random.default_rng(1)
batch = 128
EPOCHS = 20
for epoch in range(EPOCHS):
    if epoch == 14:
        opt.lr = 3e-4
    order = order_rng.permutation(len(images))
    running = []
    for s in range(0, len(images) - batch + 1, batch):
        idx = order[s:s + batch]
        cls = labels[idx].copy()
        cls[noise_rng.uniform(size=batch) < 0.1] = NULL_CLASS
        ts = noise_rng.integers(1, T + 1, batch)
        eps = noise_rng.standard_normal((batch, 784))
        l, grads = loss_and_grads(images[idx], cls, ts, eps)
        running.append(l)
        opt.step(params, grads)
    b = binned_eval()
    print(f"epoch {epoch + 1}/{EPOCHS}: train {np.mean(running):.4f} | "
          f"x0-mse low {b['low']:.4f} mid {b['mid']:.4f} high {b['high']:.4f}")

for i in range(len(net.W)):
    net.W[i] = f16_round(net.W[i]).astype(np.float64)
    net.b[i] = f16_round(net.b[i]).astype(np.float64)
CE = f16_round(CE).astype(np.float64)
b = binned_eval()
print(f"final fp16: low {b['low']:.4f} mid {b['mid']:.4f} high {b['high']:.4f}")

# conditioning diagnostic
er = np.random.default_rng(3)
ts = er.integers(40, 181, 768)
eps = er.standard_normal((768, 784))
cond = float(((net.forward(batch_inputs(test_images[:768], test_labels[:768], ts, eps))[-1] - test_images[:768]) ** 2).mean())
null = float(((net.forward(batch_inputs(test_images[:768], np.full(768, NULL_CLASS), ts, eps))[-1] - test_images[:768]) ** 2).mean())
print(f"mid-t x0-mse: conditioned {cond:.4f} vs null {null:.4f} (gain {null - cond:+.4f})")


def predict_x0(xt, ts, cls_idx):
    inp = np.concatenate([xt, t_embed(ts), CE[np.full(len(xt), cls_idx, dtype=int)]], axis=1)
    return np.clip(net.forward(inp)[-1], -1, 1)


def ddim_sample(cls_idx, n, steps=50, guidance=2.0, seed=11):
    sr = np.random.default_rng(seed + cls_idx)
    x = sr.standard_normal((n, 784))
    times = [round(T - i * T / steps) for i in range(steps)] + [0]
    x0 = None
    for i in range(steps):
        t, tp = times[i], times[i + 1]
        ts = np.full(n, t)
        sa = np.sqrt(abar[t])
        sb = np.sqrt(1 - abar[t])
        x0_c = predict_x0(x, ts, cls_idx)
        if guidance > 0:
            x0_n = predict_x0(x, ts, NULL_CLASS)
            x0 = np.clip(x0_c + guidance * (x0_c - x0_n), -1, 1)
        else:
            x0 = x0_c
        eps = (x - sa * x0) / sb
        x = np.sqrt(abar[tp]) * x0 + np.sqrt(1 - abar[tp]) * eps
    return x0


def cnn_classify(x01):
    _, w = read_nnw1(root / "public/data/weights/cnn-ref.bin")
    X = x01.reshape(-1, 1, 28, 28)

    def conv(Xc, W, bias):
        Bn, C, H, Wd = Xc.shape
        OC, _, KH, KW = W.shape
        OH, OW = H - KH + 1, Wd - KW + 1
        Z = np.zeros((Bn, OC, OH, OW))
        for ky in range(KH):
            for kx in range(KW):
                Z += np.einsum("bchw,oc->bohw", Xc[:, :, ky:ky + OH, kx:kx + OW], W[:, :, ky, kx])
        return Z + bias[None, :, None, None]

    def pool(Xc):
        Bn, C, H, Wd = Xc.shape
        OH, OW = H // 2, Wd // 2
        return Xc[:, :, :OH * 2, :OW * 2].reshape(Bn, C, OH, 2, OW, 2).max(axis=(3, 5))

    a = pool(np.maximum(conv(X, w["c1w"], w["c1b"]), 0))
    a = pool(np.maximum(conv(a, w["c2w"], w["c2b"]), 0))
    logits = a.reshape(len(X), -1) @ w["fcw"].T + w["fcb"]
    return logits.argmax(axis=1)


def ascii_digit(img01):
    chars = " .:-=+*#%@"
    rows = []
    for y in range(0, 28, 2):
        row = ""
        for x in range(28):
            v = (img01[y * 28 + x] + img01[min(27, y + 1) * 28 + x]) / 2
            row += chars[min(9, max(0, int(v * 9.99)))]
        rows.append(row)
    return rows


# export before gating so failures stay diagnosable
tensors = [(f"w{i}", net.W[i], "f16") for i in range(len(net.W))]
tensors += [(f"b{i}", net.b[i], "f16") for i in range(len(net.b))]
tensors.append(("ce", CE, "f16"))

gr = np.random.default_rng(123)
xt1 = np.tanh(np.linspace(-2, 2, 784) + 0.1 * gr.standard_normal(784))[None, :]
x0_hat = np.clip(net.forward(np.concatenate([xt1, t_embed([100]), CE[[3]]], axis=1))[-1][0], -1, 1)
fixture = {
    "xt": xt1[0].tolist(),
    "t": 100,
    "classIdx": 3,
    "x0HatHead": x0_hat[:16].tolist(),
    "tEmbedHead": t_embed([100])[0][:8].tolist(),
}

best = (0.0, -1.0)
for g in [0.0, 1.0, 2.0, 3.0]:
    hits = 0
    per_class = []
    for d in range(10):
        x0 = ddim_sample(d, 6, guidance=g)
        pred = cnn_classify((x0 + 1) / 2)
        per_class.append(int((pred == d).sum()))
        hits += int((pred == d).sum())
    agreement = hits / 60
    print(f"guidance {g}: cnn agreement {agreement:.2f} per-class {per_class}")
    if agreement > best[1]:
        best = (g, agreement)

g, agreement = best
print(f"best guidance {g}: {agreement:.2f}")
for d in [0, 3, 7]:
    x0 = ddim_sample(d, 1, guidance=g)
    print(f"--- sample of {d} (guidance {g}) ---")
    for row in ascii_digit((x0[0] + 1) / 2):
        print(row)

write_nnw1(
    root / "public/data/weights/diffusion-ref.bin",
    {"kind": "diffusion", "T": T, "tDim": TDIM, "cDim": CDIM, "hidden": "tanh",
     "predicts": "x0", "sizes": SIZES,
     "midMse": round(b["mid"], 4), "cnnAgreement": round(agreement, 3),
     "bestGuidance": g},
    tensors,
)
(root / "tests/fixtures/diffusion-golden.json").write_text(json.dumps(fixture))
print("wrote tests/fixtures/diffusion-golden.json")

assert agreement >= 0.6, "generated digits not recognizable enough to ship"
print("diffusion gates passed")
