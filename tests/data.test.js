import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../public/js/nn/rng.js';
import { datasets, circles } from '../public/js/nn/data.js';

const names = Object.keys(datasets);

test('generators return n points with binary labels', () => {
    for (const name of names) {
        const points = datasets[name](200, 0.1, mulberry32(5));
        assert.equal(points.length, 200, name);
        for (const p of points) {
            assert.ok(p.label === 0 || p.label === 1, `${name}: label ${p.label}`);
            assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y), name);
        }
    }
});

test('both classes are represented', () => {
    for (const name of names) {
        const points = datasets[name](100, 0.1, mulberry32(11));
        const ones = points.filter((p) => p.label === 1).length;
        assert.ok(ones > 20 && ones < 80, `${name}: unbalanced (${ones}/100)`);
    }
});

test('noise-free coordinates stay inside the unit-ish box', () => {
    for (const name of names) {
        const points = datasets[name](300, 0, mulberry32(3));
        for (const p of points) {
            assert.ok(Math.abs(p.x) <= 1.05 && Math.abs(p.y) <= 1.05,
                `${name}: (${p.x}, ${p.y}) out of bounds`);
        }
    }
});

test('same seed reproduces the same dataset', () => {
    for (const name of names) {
        const a = datasets[name](50, 0.2, mulberry32(21));
        const b = datasets[name](50, 0.2, mulberry32(21));
        assert.deepEqual(a, b, name);
    }
});

test('different seeds give different datasets', () => {
    for (const name of names) {
        const a = datasets[name](50, 0.2, mulberry32(1));
        const b = datasets[name](50, 0.2, mulberry32(2));
        assert.notDeepEqual(a, b, name);
    }
});

test('noise-free circles are radially separable', () => {
    const points = circles(200, 0, mulberry32(9));
    const radius = (p) => Math.hypot(p.x, p.y);
    const maxInner = Math.max(...points.filter((p) => p.label === 1).map(radius));
    const minOuter = Math.min(...points.filter((p) => p.label === 0).map(radius));
    assert.ok(maxInner < minOuter, `inner ${maxInner} overlaps outer ${minOuter}`);
});
