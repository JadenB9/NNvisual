import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../public/js/engine/rng.js';
import {
    affineForward, affineBackward, matmul, relu, reluBackward,
    gelu, softmaxRows, layernorm, argmax,
} from '../public/js/engine/tensor.js';

function randArray(n, rand) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = rand() * 2 - 1;
    return out;
}

test('affineForward matches a naive loop', () => {
    const rand = mulberry32(1);
    const batch = 3, nIn = 5, nOut = 4;
    const x = randArray(batch * nIn, rand);
    const w = randArray(nOut * nIn, rand);
    const b = randArray(nOut, rand);
    const y = affineForward(x, w, b, batch, nIn, nOut);
    for (let bi = 0; bi < batch; bi++) {
        for (let o = 0; o < nOut; o++) {
            let expected = b[o];
            for (let i = 0; i < nIn; i++) expected += x[bi * nIn + i] * w[o * nIn + i];
            assert.ok(Math.abs(y[bi * nOut + o] - expected) < 1e-5);
        }
    }
});

test('matmul matches a naive loop', () => {
    const rand = mulberry32(2);
    const m = 3, k = 4, n = 5;
    const a = randArray(m * k, rand);
    const b = randArray(k * n, rand);
    const c = matmul(a, b, m, k, n);
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            let expected = 0;
            for (let p = 0; p < k; p++) expected += a[i * k + p] * b[p * n + j];
            assert.ok(Math.abs(c[i * n + j] - expected) < 1e-5);
        }
    }
});

test('affineBackward matches numerical gradients', () => {
    const rand = mulberry32(3);
    const batch = 2, nIn = 4, nOut = 3;
    const x = randArray(batch * nIn, rand);
    const w = randArray(nOut * nIn, rand);
    const b = randArray(nOut, rand);
    const dy = randArray(batch * nOut, rand);

    // loss = sum(y * dy) so dLoss/dy = dy
    const loss = (xx, ww, bb) => {
        const y = affineForward(xx, ww, bb, batch, nIn, nOut);
        let s = 0;
        for (let i = 0; i < y.length; i++) s += y[i] * dy[i];
        return s;
    };

    const dw = new Float32Array(nOut * nIn);
    const db = new Float32Array(nOut);
    const dx = new Float32Array(batch * nIn);
    affineBackward(x, w, dy, batch, nIn, nOut, dw, db, dx);

    const h = 1e-3;
    for (let i = 0; i < w.length; i++) {
        const saved = w[i];
        w[i] = saved + h; const up = loss(x, w, b);
        w[i] = saved - h; const down = loss(x, w, b);
        w[i] = saved;
        assert.ok(Math.abs((up - down) / (2 * h) - dw[i]) < 1e-2, `dw[${i}]`);
    }
    for (let i = 0; i < x.length; i++) {
        const saved = x[i];
        x[i] = saved + h; const up = loss(x, w, b);
        x[i] = saved - h; const down = loss(x, w, b);
        x[i] = saved;
        assert.ok(Math.abs((up - down) / (2 * h) - dx[i]) < 1e-2, `dx[${i}]`);
    }
    for (let i = 0; i < b.length; i++) {
        const saved = b[i];
        b[i] = saved + h; const up = loss(x, w, b);
        b[i] = saved - h; const down = loss(x, w, b);
        b[i] = saved;
        assert.ok(Math.abs((up - down) / (2 * h) - db[i]) < 1e-2, `db[${i}]`);
    }
});

test('relu and its backward', () => {
    const y = relu(Float32Array.from([-2, 0, 3]));
    assert.deepEqual([...y], [0, 0, 3]);
    const dx = new Float32Array(3);
    reluBackward(y, Float32Array.from([5, 5, 5]), dx);
    assert.deepEqual([...dx], [0, 0, 5]);
});

test('gelu known values (tanh approximation)', () => {
    const y = gelu(Float32Array.from([0, 1, -1]));
    assert.ok(Math.abs(y[0]) < 1e-7);
    assert.ok(Math.abs(y[1] - 0.8412) < 1e-3);
    assert.ok(Math.abs(y[2] - -0.1588) < 1e-3);
});

test('softmax rows sum to 1 and survive huge logits', () => {
    const p = softmaxRows(Float32Array.from([1000, 1001, 999, 0, 0, 0]), 2, 3);
    for (let r = 0; r < 2; r++) {
        let sum = 0;
        for (let c = 0; c < 3; c++) {
            sum += p[r * 3 + c];
            assert.ok(Number.isFinite(p[r * 3 + c]));
        }
        assert.ok(Math.abs(sum - 1) < 1e-5, `row ${r} sums to ${sum}`);
    }
    assert.ok(p[1] > p[0] && p[0] > p[2]);
});

test('layernorm normalizes then scales and shifts', () => {
    const rand = mulberry32(4);
    const x = randArray(8, rand);
    const ones = new Float32Array(8).fill(1);
    const zeros = new Float32Array(8);
    const y = layernorm(x, ones, zeros, 1, 8);
    let mean = 0;
    for (const v of y) mean += v;
    mean /= 8;
    let variance = 0;
    for (const v of y) variance += (v - mean) ** 2;
    variance /= 8;
    assert.ok(Math.abs(mean) < 1e-5);
    assert.ok(Math.abs(variance - 1) < 1e-3);

    const gamma = new Float32Array(8).fill(2);
    const beta = new Float32Array(8).fill(0.5);
    const y2 = layernorm(x, gamma, beta, 1, 8);
    for (let i = 0; i < 8; i++) assert.ok(Math.abs(y2[i] - (y[i] * 2 + 0.5)) < 1e-4);
});

test('argmax with offset windows', () => {
    const x = Float32Array.from([1, 9, 2, 3, 8, 4]);
    assert.equal(argmax(x), 1);
    assert.equal(argmax(x, 3, 3), 1); // 8 at local index 1
});
