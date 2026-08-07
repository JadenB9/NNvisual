import { gaussian } from './rng.js';

// Toy 2-D datasets for binary classification. Every generator returns n
// points shaped { x, y, label } with coordinates in roughly [-1, 1] and
// label 0 or 1. `rand` is a seeded PRNG (see rng.js), so the same seed
// always produces the same dataset.

function jitter(point, noise, rand) {
    point.x += gaussian(rand) * noise;
    point.y += gaussian(rand) * noise;
    return point;
}

// A tight cluster inside a surrounding ring.
export function circles(n, noise, rand) {
    const points = [];
    for (let i = 0; i < n; i++) {
        const inner = i % 2 === 0;
        const r = inner ? rand() * 0.35 : 0.55 + rand() * 0.35;
        const t = rand() * 2 * Math.PI;
        points.push(jitter({
            x: r * Math.cos(t),
            y: r * Math.sin(t),
            label: inner ? 1 : 0,
        }, noise, rand));
    }
    return points;
}

// Two interleaved half-moons.
export function moons(n, noise, rand) {
    const points = [];
    for (let i = 0; i < n; i++) {
        const top = i % 2 === 0;
        const t = rand() * Math.PI;
        const x = top ? Math.cos(t) : 1 - Math.cos(t);
        const y = top ? Math.sin(t) : 0.5 - Math.sin(t);
        points.push(jitter({
            x: (x - 0.5) / 1.6,
            y: (y - 0.25) / 1.1,
            label: top ? 1 : 0,
        }, noise, rand));
    }
    return points;
}

// Quadrants: label follows the sign of x · y.
export function xor(n, noise, rand) {
    const points = [];
    for (let i = 0; i < n; i++) {
        // hold points off the axes so the four blocks read clearly
        const x = (0.08 + rand() * 0.92) * (rand() < 0.5 ? -1 : 1);
        const y = (0.08 + rand() * 0.92) * (rand() < 0.5 ? -1 : 1);
        points.push(jitter({ x, y, label: x * y > 0 ? 1 : 0 }, noise, rand));
    }
    return points;
}

// Two spiral arms — the classic hard case.
export function spiral(n, noise, rand) {
    const points = [];
    for (let i = 0; i < n; i++) {
        const arm = i % 2;
        const frac = rand();
        const r = 0.08 + frac * 0.8;
        const t = frac * 3.2 * Math.PI + arm * Math.PI;
        points.push(jitter({
            x: r * Math.sin(t),
            y: r * Math.cos(t),
            label: arm,
        }, noise * 0.5, rand));
    }
    return points;
}

export const datasets = { circles, moons, xor, spiral };
