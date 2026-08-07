import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../public/js/engine/rng.js';
import { Transformer } from '../public/js/engine/transformer.js';

// Small random model built the way parseWeights would return it.
function randomModel(seed) {
    const vocab = 11;
    const block = 8;
    const d = 16;
    const nHead = 2;
    const nLayer = 2;
    const rand = mulberry32(seed);
    const mk = (n, scale = 0.3) => {
        const a = new Float32Array(n);
        for (let i = 0; i < n; i++) a[i] = (rand() * 2 - 1) * scale;
        return a;
    };
    const ones = (n) => new Float32Array(n).fill(1);
    const zeros = (n) => new Float32Array(n);

    const tensors = {
        tok_emb: { shape: [vocab, d], data: mk(vocab * d) },
        pos_emb: { shape: [block, d], data: mk(block * d) },
        lnf_g: { shape: [d], data: ones(d) },
        lnf_b: { shape: [d], data: zeros(d) },
        head: { shape: [vocab, d], data: mk(vocab * d) },
    };
    for (let l = 0; l < nLayer; l++) {
        tensors[`l${l}_ln1_g`] = { shape: [d], data: ones(d) };
        tensors[`l${l}_ln1_b`] = { shape: [d], data: zeros(d) };
        tensors[`l${l}_ln2_g`] = { shape: [d], data: ones(d) };
        tensors[`l${l}_ln2_b`] = { shape: [d], data: zeros(d) };
        for (const n of ['wq', 'wk', 'wv', 'wo']) tensors[`l${l}_${n}`] = { shape: [d, d], data: mk(d * d) };
        for (const n of ['bq', 'bk', 'bv', 'bo']) tensors[`l${l}_${n}`] = { shape: [d], data: zeros(d) };
        tensors[`l${l}_fc1`] = { shape: [4 * d, d], data: mk(4 * d * d) };
        tensors[`l${l}_fc1b`] = { shape: [4 * d], data: zeros(4 * d) };
        tensors[`l${l}_fc2`] = { shape: [d, 4 * d], data: mk(d * 4 * d) };
        tensors[`l${l}_fc2b`] = { shape: [d], data: zeros(d) };
    }
    const chars = 'abcdefghij\n'.split('');
    return new Transformer({
        meta: { vocab, block, dModel: d, nHead, nLayer, chars },
        tensors,
    });
}

test('encode/decode round-trip, unknown chars dropped', () => {
    const m = randomModel(1);
    const toks = m.encode('abcz\n');
    assert.deepEqual(toks, [0, 1, 2, 10]);
    assert.equal(m.decode(toks), 'abc\n');
});

test('forward captures per-layer, per-head causal attention', () => {
    const m = randomModel(2);
    const fwd = m.forward([0, 1, 2, 3, 4]);
    assert.equal(fwd.layers.length, 2);
    assert.equal(fwd.layers[0].heads.length, 2);
    const probs = fwd.layers[0].heads[0].probs;
    // rows sum to 1; future positions carry zero weight
    for (let i = 0; i < 5; i++) {
        let sum = 0;
        for (let j = 0; j < 5; j++) {
            sum += probs[i * 5 + j];
            if (j > i) assert.ok(probs[i * 5 + j] < 1e-9, `future leak at ${i},${j}`);
        }
        assert.ok(Math.abs(sum - 1) < 1e-4);
    }
});

test('causality: changing a later token never changes earlier logits', () => {
    const m = randomModel(3);
    const a = m.forward([0, 1, 2, 3]);
    const b = m.forward([0, 1, 2, 9]);
    for (let pos = 0; pos < 3; pos++) {
        for (let v = 0; v < 11; v++) {
            assert.ok(
                Math.abs(a.logits[pos * 11 + v] - b.logits[pos * 11 + v]) < 1e-4,
                `pos ${pos} vocab ${v}`
            );
        }
    }
});

test('temperature sharpens and top-k truncates the distribution', () => {
    const m = randomModel(4);
    const ctx = [0, 1, 2];
    const hot = m.nextDistribution(ctx, 2).probs;
    const cold = m.nextDistribution(ctx, 0.25).probs;
    const maxOf = (p) => Math.max(...p);
    assert.ok(maxOf(cold) > maxOf(hot));

    const k2 = m.nextDistribution(ctx, 1, 2).probs;
    let nonzero = 0;
    let sum = 0;
    for (const p of k2) { if (p > 1e-9) nonzero++; sum += p; }
    assert.equal(nonzero, 2);
    assert.ok(Math.abs(sum - 1) < 1e-5);
});

test('sampling follows the distribution deterministically with a seed', () => {
    const m = randomModel(5);
    const { probs } = m.nextDistribution([0, 1], 1);
    const rand = mulberry32(9);
    const first = m.sample(probs, rand);
    const again = m.sample(probs, mulberry32(9));
    assert.equal(first, again);
    assert.ok(first >= 0 && first < 11);
});
