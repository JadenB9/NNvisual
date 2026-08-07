import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, gaussian, fillGaussian, shuffle } from '../public/js/engine/rng.js';

test('same seed gives the same sequence', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 100; i++) assert.equal(a(), b());
});

test('values stay in [0, 1)', () => {
    const rand = mulberry32(7);
    for (let i = 0; i < 1000; i++) {
        const v = rand();
        assert.ok(v >= 0 && v < 1);
    }
});

test('gaussian is roughly standard', () => {
    const rand = mulberry32(99);
    const n = 5000;
    let sum = 0, sumSq = 0;
    for (let i = 0; i < n; i++) {
        const g = gaussian(rand);
        sum += g;
        sumSq += g * g;
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    assert.ok(Math.abs(mean) < 0.1);
    assert.ok(Math.abs(variance - 1) < 0.15);
});

test('fillGaussian fills every slot', () => {
    const out = fillGaussian(new Float32Array(64), mulberry32(5));
    let nonzero = 0;
    for (const v of out) if (v !== 0) nonzero++;
    assert.ok(nonzero > 60);
});

test('shuffle permutes deterministically and keeps all elements', () => {
    const a = shuffle([...Array(50).keys()], mulberry32(8));
    const b = shuffle([...Array(50).keys()], mulberry32(8));
    assert.deepEqual(a, b);
    assert.deepEqual([...a].sort((x, y) => x - y), [...Array(50).keys()]);
    assert.notDeepEqual(a, [...Array(50).keys()]);
});
