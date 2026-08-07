import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, gaussian } from '../public/js/nn/rng.js';

test('same seed gives the same sequence', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 100; i++) assert.equal(a(), b());
});

test('different seeds diverge', () => {
    const a = mulberry32(1);
    const b = mulberry32(2);
    const seqA = Array.from({ length: 10 }, a);
    const seqB = Array.from({ length: 10 }, b);
    assert.notDeepEqual(seqA, seqB);
});

test('values stay in [0, 1)', () => {
    const rand = mulberry32(7);
    for (let i = 0; i < 1000; i++) {
        const v = rand();
        assert.ok(v >= 0 && v < 1, `out of range: ${v}`);
    }
});

test('gaussian is roughly centered with unit-ish spread', () => {
    const rand = mulberry32(99);
    const n = 5000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
        const g = gaussian(rand);
        sum += g;
        sumSq += g * g;
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    assert.ok(Math.abs(mean) < 0.1, `mean drifted: ${mean}`);
    assert.ok(Math.abs(variance - 1) < 0.15, `variance drifted: ${variance}`);
});
