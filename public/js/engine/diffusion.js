import { fillGaussian, mulberry32 } from './rng.js';

// DDPM math on 28x28 images in [-1, 1], with a DDIM sampler that emits
// every intermediate so the page can show the whole trajectory.

// Cosine noise schedule (Nichol & Dhariwal). alphaBar[t] for t = 0..T,
// alphaBar[0] = 1.
export function makeSchedule(T, s = 0.008) {
    const alphaBar = new Float32Array(T + 1);
    const f = (t) => Math.cos(((t / T + s) / (1 + s)) * Math.PI / 2) ** 2;
    const f0 = f(0);
    for (let t = 0; t <= T; t++) alphaBar[t] = Math.min(1, Math.max(1e-5, f(t) / f0));
    return alphaBar;
}

// x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps
export function forwardNoise(x0, eps, abar, out) {
    out = out || new Float32Array(x0.length);
    const a = Math.sqrt(abar);
    const b = Math.sqrt(1 - abar);
    for (let i = 0; i < x0.length; i++) out[i] = a * x0[i] + b * eps[i];
    return out;
}

// Sinusoidal step embedding; layout [sin(t*f_0..f_{h-1}), cos(...)] with
// f_i = 10000^(-i/h). Must match training/common.py exactly.
export function tEmbed(t, dim, out) {
    out = out || new Float32Array(dim);
    const half = dim / 2;
    for (let i = 0; i < half; i++) {
        const f = Math.pow(10000, -i / half);
        out[i] = Math.sin(t * f);
        out[half + i] = Math.cos(t * f);
    }
    return out;
}

// Step-at-a-time DDIM sampler (eta = 0) with classifier-free guidance.
// `predict(xt, t, classIdx)` -> eps-hat; classIdx 10 is the null class.
// Returns an object the page drives one step per frame.
export function createSampler(predict, { T = 200, steps = 30, classIdx = 0, guidance = 2, seed = 1 }) {
    const alphaBar = makeSchedule(T);
    const rand = mulberry32(seed);
    const x = fillGaussian(new Float32Array(784), rand);

    // step times T -> ~0, evenly spaced
    const times = [];
    for (let i = 0; i < steps; i++) times.push(Math.round(T - (i * T) / steps));
    times.push(0);

    let at = 0;
    const history = [];

    return {
        history,
        get done() { return at >= steps; },
        step() {
            if (at >= steps) return null;
            const t = times[at];
            const tPrev = times[at + 1];
            const abarT = alphaBar[t];
            const abarPrev = alphaBar[tPrev];

            // classifier-free guidance: push along (conditional - unconditional)
            const epsC = predict(x, t, classIdx);
            let eps = epsC;
            if (guidance > 0 && classIdx !== 10) {
                const epsNull = predict(x, t, 10);
                eps = new Float32Array(784);
                for (let i = 0; i < 784; i++) {
                    eps[i] = epsC[i] + guidance * (epsC[i] - epsNull[i]);
                }
            }

            const x0 = new Float32Array(784);
            const sa = Math.sqrt(abarT);
            const sb = Math.sqrt(1 - abarT);
            for (let i = 0; i < 784; i++) {
                x0[i] = Math.min(1, Math.max(-1, (x[i] - sb * eps[i]) / sa));
            }

            // Re-derive the noise from the clamped image before stepping.
            // Otherwise a guided prediction that clamps keeps its over-shot
            // eps and the excess compounds every step into saturation.
            if (sb > 1e-4) {
                for (let i = 0; i < 784; i++) eps[i] = (x[i] - sa * x0[i]) / sb;
            }

            const saP = Math.sqrt(abarPrev);
            const sbP = Math.sqrt(1 - abarPrev);
            for (let i = 0; i < 784; i++) x[i] = saP * x0[i] + sbP * eps[i];

            const record = { t, tPrev, xt: x.slice(), epsHat: eps.slice(), x0Hat: x0 };
            history.push(record);
            at++;
            return record;
        },
    };
}
