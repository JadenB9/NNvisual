// Deterministic PRNG (mulberry32). Everything random in the app — weight
// init, dataset sampling, noise — flows through one of these, so a given
// seed reproduces a run exactly. The tests depend on that too.
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

// Standard normal via Box–Muller, driven by a seeded rand.
export function gaussian(rand) {
    let u = 0;
    while (u === 0) u = rand(); // log(0) guard
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}
