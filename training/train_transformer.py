"""Char-level GPT on tiny-shakespeare, trained with hand-written numpy
backprop (gradient-checked on a tiny config before the real run).

Architecture — must mirror js/engine/transformer.js exactly:
  x = tok_emb[ids] + pos_emb[:T]
  per layer (pre-LN): x += Wo(attn(LN1(x))); x += fc2(gelu(fc1(LN2(x))))
  logits = LNf(x) @ head.T           (gelu is the tanh approximation)

Exports public/data/weights/transformer-ref.bin (fp16) with the vocab in
meta, plus a golden logits fixture computed from the rounded weights."""

import json
from pathlib import Path

import numpy as np

from common import Adam, write_nnw1, f16_round

root = Path(__file__).resolve().parent.parent
GELU_C = np.sqrt(2 / np.pi)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(GELU_C * (x + 0.044715 * x ** 3)))


def gelu_bwd(x, dy):
    u = GELU_C * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GELU_C * (1 + 3 * 0.044715 * x * x))


def ln_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1 / np.sqrt(var + 1e-5)
    xhat = (x - mu) * inv
    return xhat * g + b, (xhat, inv)


def ln_bwd(dy, g, cache):
    xhat, inv = cache
    n = xhat.shape[-1]
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    db = dy.sum(axis=tuple(range(dy.ndim - 1)))
    dxhat = dy * g
    dx = inv * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def linear_fwd(x, W, b):
    return x @ W.T + (b if b is not None else 0)


def linear_bwd(x, W, dy):
    flat_x = x.reshape(-1, x.shape[-1])
    flat_dy = dy.reshape(-1, dy.shape[-1])
    dW = flat_dy.T @ flat_x
    db = flat_dy.sum(0)
    dx = dy @ W
    return dW, db, dx


class GPT:
    def __init__(self, vocab, block, d, n_head, n_layer, seed=0):
        rng = np.random.default_rng(seed)
        self.cfg = dict(vocab=vocab, block=block, d=d, n_head=n_head, n_layer=n_layer)
        p = {}
        p["tok_emb"] = rng.normal(0, 0.02, (vocab, d))
        p["pos_emb"] = rng.normal(0, 0.02, (block, d))
        for l in range(n_layer):
            for n in ["wq", "wk", "wv", "wo"]:
                p[f"l{l}_{n}"] = rng.normal(0, 0.02, (d, d))
            for n in ["bq", "bk", "bv", "bo"]:
                p[f"l{l}_{n}"] = np.zeros(d)
            p[f"l{l}_ln1_g"] = np.ones(d)
            p[f"l{l}_ln1_b"] = np.zeros(d)
            p[f"l{l}_ln2_g"] = np.ones(d)
            p[f"l{l}_ln2_b"] = np.zeros(d)
            p[f"l{l}_fc1"] = rng.normal(0, 0.02, (4 * d, d))
            p[f"l{l}_fc1b"] = np.zeros(4 * d)
            p[f"l{l}_fc2"] = rng.normal(0, 0.02 / np.sqrt(2 * n_layer), (d, 4 * d))
            p[f"l{l}_fc2b"] = np.zeros(d)
        p["lnf_g"] = np.ones(d)
        p["lnf_b"] = np.zeros(d)
        p["head"] = rng.normal(0, 0.02, (vocab, d))
        self.p = p

    def forward(self, ids):
        p = self.p
        cfg = self.cfg
        B, T = ids.shape
        d, H, L = cfg["d"], cfg["n_head"], cfg["n_layer"]
        dh = d // H
        cache = {"ids": ids, "layers": []}
        x = p["tok_emb"][ids] + p["pos_emb"][:T]
        mask = np.tril(np.ones((T, T), dtype=bool))
        for l in range(L):
            c = {}
            c["x_in"] = x
            h, c["ln1"] = ln_fwd(x, p[f"l{l}_ln1_g"], p[f"l{l}_ln1_b"])
            c["h"] = h
            q = linear_fwd(h, p[f"l{l}_wq"], p[f"l{l}_bq"])
            k = linear_fwd(h, p[f"l{l}_wk"], p[f"l{l}_bk"])
            v = linear_fwd(h, p[f"l{l}_wv"], p[f"l{l}_bv"])
            # (B,H,T,dh)
            def split(z):
                return z.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            Q, K, V = split(q), split(k), split(v)
            S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh)
            S = np.where(mask, S, -1e30)
            P = np.exp(S - S.max(-1, keepdims=True))
            P /= P.sum(-1, keepdims=True)
            O = P @ V
            merged = O.transpose(0, 2, 1, 3).reshape(B, T, d)
            attn_out = linear_fwd(merged, p[f"l{l}_wo"], p[f"l{l}_bo"])
            x = x + attn_out
            c.update(q=q, k=k, v=v, Q=Q, K=K, V=V, P=P, merged=merged)
            c["x_mid"] = x
            h2, c["ln2"] = ln_fwd(x, p[f"l{l}_ln2_g"], p[f"l{l}_ln2_b"])
            c["h2"] = h2
            z1 = linear_fwd(h2, p[f"l{l}_fc1"], p[f"l{l}_fc1b"])
            a1 = gelu(z1)
            m = linear_fwd(a1, p[f"l{l}_fc2"], p[f"l{l}_fc2b"])
            x = x + m
            c.update(z1=z1, a1=a1)
            cache["layers"].append(c)
        xf, cache["lnf"] = ln_fwd(x, p["lnf_g"], p["lnf_b"])
        cache["x_last"] = x
        cache["xf"] = xf
        logits = xf @ p["head"].T
        return logits, cache

    def loss_and_grads(self, ids, targets):
        p = self.p
        cfg = self.cfg
        B, T = ids.shape
        d, H, L = cfg["d"], cfg["n_head"], cfg["n_layer"]
        dh = d // H
        logits, cache = self.forward(ids)

        m = logits - logits.max(-1, keepdims=True)
        e = np.exp(m)
        probs = e / e.sum(-1, keepdims=True)
        n = B * T
        loss = -np.log(np.clip(probs[np.arange(B)[:, None], np.arange(T), targets], 1e-12, None)).mean()
        dlogits = probs.copy()
        dlogits[np.arange(B)[:, None], np.arange(T), targets] -= 1
        dlogits /= n

        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["head"] = dlogits.reshape(-1, cfg["vocab"]).T @ cache["xf"].reshape(-1, d)
        dxf = dlogits @ p["head"]
        dx, g["lnf_g"], g["lnf_b"] = ln_bwd(dxf, p["lnf_g"], cache["lnf"])

        for l in reversed(range(L)):
            c = cache["layers"][l]
            # mlp branch
            dm = dx
            dW2, db2, da1 = linear_bwd(c["a1"], p[f"l{l}_fc2"], dm)
            g[f"l{l}_fc2"] += dW2
            g[f"l{l}_fc2b"] += db2
            dz1 = gelu_bwd(c["z1"], da1)
            dW1, db1, dh2 = linear_bwd(c["h2"], p[f"l{l}_fc1"], dz1)
            g[f"l{l}_fc1"] += dW1
            g[f"l{l}_fc1b"] += db1
            dx_mid, dg2, dbt2 = ln_bwd(dh2, p[f"l{l}_ln2_g"], c["ln2"])
            g[f"l{l}_ln2_g"] += dg2
            g[f"l{l}_ln2_b"] += dbt2
            dx = dx + dx_mid  # residual

            # attention branch
            dattn = dx
            dWo, dbo, dmerged = linear_bwd(c["merged"], p[f"l{l}_wo"], dattn)
            g[f"l{l}_wo"] += dWo
            g[f"l{l}_bo"] += dbo
            dO = dmerged.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            dP = dO @ c["V"].transpose(0, 1, 3, 2)
            dV = c["P"].transpose(0, 1, 3, 2) @ dO
            P = c["P"]
            dS = P * (dP - (dP * P).sum(-1, keepdims=True))
            dQ = dS @ c["K"] / np.sqrt(dh)
            dK = dS.transpose(0, 1, 3, 2) @ c["Q"] / np.sqrt(dh)

            def merge(z):
                return z.transpose(0, 2, 1, 3).reshape(B, T, d)
            dq, dk, dv = merge(dQ), merge(dK), merge(dV)
            dWq, dbq, dh_q = linear_bwd(c["h"], p[f"l{l}_wq"], dq)
            dWk, dbk, dh_k = linear_bwd(c["h"], p[f"l{l}_wk"], dk)
            dWv, dbv, dh_v = linear_bwd(c["h"], p[f"l{l}_wv"], dv)
            g[f"l{l}_wq"] += dWq
            g[f"l{l}_bq"] += dbq
            g[f"l{l}_wk"] += dWk
            g[f"l{l}_bk"] += dbk
            g[f"l{l}_wv"] += dWv
            g[f"l{l}_bv"] += dbv
            dh_sum = dh_q + dh_k + dh_v
            dx_in, dg1, dbt1 = ln_bwd(dh_sum, p[f"l{l}_ln1_g"], c["ln1"])
            g[f"l{l}_ln1_g"] += dg1
            g[f"l{l}_ln1_b"] += dbt1
            dx = dx + dx_in  # residual

        np.add.at(g["tok_emb"], cache["ids"], dx)
        g["pos_emb"][:T] += dx.sum(0)
        return loss, g


def grad_check_tiny():
    rng = np.random.default_rng(5)
    m = GPT(vocab=11, block=6, d=8, n_head=2, n_layer=1, seed=9)
    ids = rng.integers(0, 11, (2, 6))
    targets = rng.integers(0, 11, (2, 6))
    loss, g = m.loss_and_grads(ids, targets)
    h = 1e-5
    for name in m.p:
        flat_p = m.p[name].reshape(-1)
        flat_g = g[name].reshape(-1)
        for _ in range(4):
            i = rng.integers(flat_p.size)
            saved = flat_p[i]
            flat_p[i] = saved + h
            up = m.loss_and_grads(ids, targets)[0]
            flat_p[i] = saved - h
            down = m.loss_and_grads(ids, targets)[0]
            flat_p[i] = saved
            numeric = (up - down) / (2 * h)
            denom = max(abs(numeric), abs(flat_g[i]), 1e-8)
            assert abs(numeric - flat_g[i]) / denom < 1e-3 or abs(numeric - flat_g[i]) < 1e-8, \
                f"{name}[{i}]: numeric {numeric} vs analytic {flat_g[i]}"
    print("transformer grad check ok")


grad_check_tiny()

# ---------- real training ----------

text = (Path(__file__).parent / "data/shakespeare.txt").read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int32)
split = int(len(data) * 0.95)
train, val = data[:split], data[split:]
V = len(chars)
print(f"corpus {len(data)} chars, vocab {V}")

BLOCK, D, HEADS, LAYERS = 64, 64, 4, 2
BATCH, STEPS = 32, 4000
model = GPT(V, BLOCK, D, HEADS, LAYERS, seed=1)
opt = Adam(list(model.p.values()), lr=1e-3)
names = list(model.p.keys())
rng = np.random.default_rng(0)


def get_batch(src, n):
    ix = rng.integers(0, len(src) - BLOCK - 1, n)
    x = np.stack([src[i:i + BLOCK] for i in ix])
    y = np.stack([src[i + 1:i + 1 + BLOCK] for i in ix])
    return x, y


def val_loss():
    losses = []
    for _ in range(8):
        x, y = get_batch(val, 32)
        losses.append(model.loss_and_grads(x, y)[0])
    return float(np.mean(losses))


for step in range(1, STEPS + 1):
    warm = min(1.0, step / 100)
    decay = 0.5 * (1 + np.cos(np.pi * step / STEPS))
    opt.lr = (1e-4 + 9e-4 * decay) * warm
    x, y = get_batch(train, BATCH)
    loss, g = model.loss_and_grads(x, y)
    opt.step([model.p[n] for n in names], [g[n] for n in names])
    if step % 250 == 0:
        print(f"step {step}: train {loss:.3f} val {val_loss():.3f} lr {opt.lr:.2e}")

# fp16-round shipped weights, re-measure, gate
for n in names:
    model.p[n] = f16_round(model.p[n]).astype(np.float64)
vl = val_loss()
print(f"final fp16 val loss {vl:.3f}")
assert vl < 2.05, "transformer val loss too high to ship"

# sample to eyeball quality
def sample(prompt, n=200, temp=0.8):
    ids = [stoi[c] for c in prompt]
    for _ in range(n):
        ctx = np.array([ids[-BLOCK:]])
        logits, _ = model.forward(ctx)
        z = logits[0, -1] / temp
        pz = np.exp(z - z.max())
        pz /= pz.sum()
        ids.append(int(rng.choice(V, p=pz)))
    return "".join(chars[i] for i in ids)

print(sample("ROMEO:", 300))

meta = {"kind": "transformer", "vocab": V, "block": BLOCK, "dModel": D,
        "nHead": HEADS, "nLayer": LAYERS, "chars": chars, "valLoss": round(vl, 3)}
write_nnw1(root / "public/data/weights/transformer-ref.bin", meta,
           [(n, model.p[n], "f16") for n in names])

golden_tokens = [stoi[c] for c in "ROMEO: the"]
logits, _ = model.forward(np.array([golden_tokens]))
fixture = {"tokens": golden_tokens, "lastLogits": logits[0, -1].tolist()}
(root / "tests/fixtures/transformer-golden.json").write_text(json.dumps(fixture))
print("wrote tests/fixtures/transformer-golden.json")
