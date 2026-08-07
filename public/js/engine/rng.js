// Deterministic PRNG (mulberry32). Weight init, batching order, and sampled
// noise all flow through one of these, so a given seed reproduces a run
// exactly — in the page and in the tests.
export function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Standard normal via Box–Muller.
export function gaussian(rand) {
    let u = 0;
    while (u === 0) u = rand(); // log(0) guard
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

// Fill a Float32Array with N(0, 1) noise.
export function fillGaussian(out, rand) {
    for (let i = 0; i < out.length; i++) out[i] = gaussian(rand);
    return out;
}

// Fisher–Yates shuffle of an index array.
export function shuffle(indices, rand) {
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        const t = indices[i];
        indices[i] = indices[j];
        indices[j] = t;
    }
    return indices;
}
