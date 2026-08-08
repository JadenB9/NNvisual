import { makeSchedule } from '../engine/diffusion.js';

// img2img: the same DDIM sampler as createSampler in js/engine/diffusion.js,
// except the trajectory starts at t = tStart from a supplied noised image
// instead of at t = T from pure noise. The per-step update below is a faithful
// copy of the engine's — if that one changes, this must change with it.
export function createImageSampler(predict, { T = 200, tStart = 120, steps = 30, classIdx = 0, guidance = 2, xStart = null }) {
    const alphaBar = makeSchedule(T);
    const x = xStart ? xStart.slice() : new Float32Array(784);

    // step times tStart -> ~0, evenly spaced
    const times = [];
    for (let i = 0; i < steps; i++) times.push(Math.round(tStart - (i * tStart) / steps));
    times.push(0);

    let at = 0;
    const history = [];

    return {
        history,
        tStart,
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
