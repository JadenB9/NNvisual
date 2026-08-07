import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../public/js/engine/rng.js';
import { conv2dForward, maxpool2x2, ConvNet } from '../public/js/engine/conv.js';

function randArray(n, rand) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = rand() * 2 - 1;
    return out;
}

// independent reference with a different loop order
function convRef(input, kernels, bias, C, H, W, OC, KH, KW) {
    const OH = H - KH + 1;
    const OW = W - KW + 1;
    const out = new Float32Array(OC * OH * OW);
    for (let oc = 0; oc < OC; oc++) {
        for (let oy = 0; oy < OH; oy++) {
            for (let ox = 0; ox < OW; ox++) {
                let s = bias[oc];
                for (let c = 0; c < C; c++) {
                    for (let ky = 0; ky < KH; ky++) {
                        for (let kx = 0; kx < KW; kx++) {
                            s += input[c * H * W + (oy + ky) * W + (ox + kx)]
                                * kernels[((oc * C + c) * KH + ky) * KW + kx];
                        }
                    }
                }
                out[oc * OH * OW + oy * OW + ox] = s;
            }
        }
    }
    return out;
}

test('conv2dForward matches the reference on random tensors', () => {
    const rand = mulberry32(5);
    const C = 3, H = 7, W = 6, OC = 4, KH = 3, KW = 3;
    const input = randArray(C * H * W, rand);
    const kernels = randArray(OC * C * KH * KW, rand);
    const bias = randArray(OC, rand);
    const got = conv2dForward(input, kernels, bias, C, H, W, OC, KH, KW);
    const want = convRef(input, kernels, bias, C, H, W, OC, KH, KW);
    assert.equal(got.length, want.length);
    for (let i = 0; i < got.length; i++) {
        assert.ok(Math.abs(got[i] - want[i]) < 1e-4, `at ${i}`);
    }
});

test('identity kernel reproduces the input interior', () => {
    const rand = mulberry32(6);
    const input = randArray(25, rand); // 1x5x5
    const kernel = Float32Array.from([0, 0, 0, 0, 1, 0, 0, 0, 0]);
    const out = conv2dForward(input, kernel, new Float32Array(1), 1, 5, 5, 1, 3, 3);
    for (let y = 0; y < 3; y++) {
        for (let x = 0; x < 3; x++) {
            assert.ok(Math.abs(out[y * 3 + x] - input[(y + 1) * 5 + (x + 1)]) < 1e-6);
        }
    }
});

test('maxpool takes the max and remembers where it was', () => {
    const input = Float32Array.from([
        1, 5, 2, 0,
        3, 4, 8, 1,
        0, 0, 9, 2,
        7, 6, 3, 4,
    ]); // 1x4x4
    const { out, idx } = maxpool2x2(input, 1, 4, 4);
    assert.deepEqual([...out], [5, 8, 7, 9]);
    assert.deepEqual([...idx], [1, 6, 12, 10]);
});

test('maxpool floors odd sizes', () => {
    const input = new Float32Array(1 * 5 * 5).fill(1);
    const { out } = maxpool2x2(input, 1, 5, 5);
    assert.equal(out.length, 4); // 2x2
});

test('ConvNet forward produces coherent stage shapes and a distribution', () => {
    const rand = mulberry32(77);
    const tensors = {
        c1w: { shape: [8, 1, 3, 3], data: randArray(72, rand) },
        c1b: { shape: [8], data: randArray(8, rand) },
        c2w: { shape: [16, 8, 3, 3], data: randArray(1152, rand) },
        c2b: { shape: [16], data: randArray(16, rand) },
        fcw: { shape: [10, 400], data: randArray(4000, rand) },
        fcb: { shape: [10], data: randArray(10, rand) },
    };
    const net = new ConvNet().loadTensors(tensors);
    const x = randArray(784, rand);
    for (let i = 0; i < x.length; i++) x[i] = Math.abs(x[i]);
    const s = net.forward(x);
    assert.equal(s.conv1.length, 8 * 26 * 26);
    assert.equal(s.pool1.out.length, 8 * 13 * 13);
    assert.equal(s.conv2.length, 16 * 11 * 11);
    assert.equal(s.pool2.out.length, 16 * 5 * 5);
    assert.equal(s.flat.length, 400);
    assert.equal(s.probs.length, 10);
    let sum = 0;
    for (const p of s.probs) sum += p;
    assert.ok(Math.abs(sum - 1) < 1e-5);
});
