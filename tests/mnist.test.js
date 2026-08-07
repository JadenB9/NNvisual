import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseDataset, preprocessDrawing, resampleBilinear } from '../public/js/engine/mnist.js';

function buildDataset(images, labels) {
    const n = labels.length;
    const buf = new ArrayBuffer(8 + n + n * 784);
    const u8 = new Uint8Array(buf);
    u8.set([78, 78, 68, 49], 0); // "NND1"
    new DataView(buf).setUint32(4, n, true);
    u8.set(labels, 8);
    for (let i = 0; i < n; i++) u8.set(images[i], 8 + n + i * 784);
    return buf;
}

test('parses a packed dataset', () => {
    const img0 = new Uint8Array(784);
    img0[0] = 255;
    const img1 = new Uint8Array(784).fill(128);
    const ds = parseDataset(buildDataset([img0, img1], Uint8Array.from([3, 7])));
    assert.equal(ds.count, 2);
    assert.equal(ds.labels[0], 3);
    assert.equal(ds.labels[1], 7);
    const x = ds.image(0);
    assert.equal(x[0], 1);
    assert.equal(x[1], 0);
    assert.ok(Math.abs(ds.image(1)[100] - 128 / 255) < 1e-6);
});

test('fillBatch normalizes and labels', () => {
    const img = new Uint8Array(784).fill(51); // 0.2
    const ds = parseDataset(buildDataset([img], Uint8Array.from([5])));
    const x = new Float32Array(784);
    const y = new Uint8Array(1);
    ds.fillBatch([0], 0, 1, x, y);
    assert.equal(y[0], 5);
    assert.ok(Math.abs(x[0] - 0.2) < 1e-6);
});

test('rejects wrong magic', () => {
    const buf = new ArrayBuffer(16);
    assert.throws(() => parseDataset(buf), /magic/);
});

test('bilinear resample preserves constant blocks and averages edges', () => {
    const src = Float32Array.from([1, 0, 1, 0]); // 2x2: left column 1
    const up = resampleBilinear(src, 2, 2, 4, 4);
    assert.equal(up.length, 16);
    for (const v of up) assert.ok(v >= 0 && v <= 1.0001);
    // left side should stay brighter than right
    assert.ok(up[0] > up[3]);
});

test('preprocess centers an off-corner blob at the center of mass', () => {
    const src = new Float32Array(784);
    for (let y = 0; y < 6; y++) {
        for (let x = 0; x < 6; x++) src[y * 28 + x] = 1; // blob in top-left
    }
    const out = preprocessDrawing(src);

    let mass = 0, mx = 0, my = 0;
    let x0 = 28, x1 = -1, y0 = 28, y1 = -1;
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            const v = out[y * 28 + x];
            if (v > 0.05) {
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
            }
            mass += v;
            mx += v * x;
            my += v * y;
        }
    }
    assert.ok(mass > 0);
    assert.ok(Math.abs(mx / mass - 13.5) < 1, `com x ${mx / mass}`);
    assert.ok(Math.abs(my / mass - 13.5) < 1, `com y ${my / mass}`);
    // longest side scaled to ~20
    assert.ok(Math.max(x1 - x0 + 1, y1 - y0 + 1) >= 18);
    assert.ok(Math.max(x1 - x0 + 1, y1 - y0 + 1) <= 21);
    for (const v of out) assert.ok(v >= 0 && v <= 1.0001);
});

test('preprocess of a blank canvas is all zeros', () => {
    const out = preprocessDrawing(new Float32Array(784));
    for (const v of out) assert.equal(v, 0);
});
