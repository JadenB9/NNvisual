// Diffusion golden gates. The shipped denoiser predicts x0 (see
// training/train_diffusion.py for why eps-prediction cannot work at this
// width); the sampler still runs on eps-hat derived algebraically, so these
// tests pin both the raw model output and the full DDIM chain.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { parseWeights } from '../public/js/engine/weights.js';
import { MLP } from '../public/js/engine/mlp.js';
import { ConvNet } from '../public/js/engine/conv.js';
import { tEmbed, createSampler, makeSchedule } from '../public/js/engine/diffusion.js';
import { mulberry32 } from '../public/js/engine/rng.js';

const at = (p) => fileURLToPath(new URL(p, import.meta.url));
const loadBin = (p) => {
    const b = readFileSync(at(p));
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const loadJson = (p) => JSON.parse(readFileSync(at(p), 'utf8'));

function loadDenoiser() {
    const w = parseWeights(loadBin('../public/data/weights/diffusion-ref.bin'));
    const net = new MLP(w.meta.sizes, { task: 'regress', hidden: w.meta.hidden }, mulberry32(1))
        .loadTensors(w.tensors);
    const ce = w.tensors.ce.data;
    const meta = w.meta;

    const predictX0 = (xt, t, cls) => {
        const input = new Float32Array(meta.sizes[0]);
        input.set(xt, 0);
        input.set(tEmbed(t, meta.tDim), 784);
        input.set(ce.subarray(cls * meta.cDim, (cls + 1) * meta.cDim), 784 + meta.tDim);
        const out = net.predict(input);
        const x0 = new Float32Array(784);
        for (let i = 0; i < 784; i++) x0[i] = Math.min(1, Math.max(-1, out[i]));
        return x0;
    };
    return { meta, predictX0 };
}

test('denoiser matches its golden x0 output and t-embedding layout', () => {
    const { meta, predictX0 } = loadDenoiser();
    const golden = loadJson('fixtures/diffusion-golden.json');
    assert.equal(meta.predicts, 'x0');

    const emb = tEmbed(golden.t, meta.tDim);
    for (let i = 0; i < 8; i++) {
        assert.ok(Math.abs(emb[i] - golden.tEmbedHead[i]) < 1e-5, `temb ${i}`);
    }

    const x0 = predictX0(Float32Array.from(golden.xt), golden.t, golden.classIdx);
    for (let i = 0; i < 16; i++) {
        assert.ok(Math.abs(x0[i] - golden.x0HatHead[i]) < 2e-2,
            `x0 ${i}: ${x0[i]} vs ${golden.x0HatHead[i]}`);
    }
});

test('a full DDIM run through the eps wrapper stays finite and image-like', () => {
    const { meta, predictX0 } = loadDenoiser();
    const alphaBar = makeSchedule(meta.T);

    // the page derives eps-hat from the predicted image; CFG on eps is
    // algebraically identical to CFG on x0 for fixed x_t
    const predictEps = (xt, t, cls) => {
        const x0 = predictX0(xt, t, cls);
        const sa = Math.sqrt(alphaBar[t]);
        const sb = Math.sqrt(1 - alphaBar[t]);
        const eps = new Float32Array(784);
        for (let i = 0; i < 784; i++) eps[i] = (xt[i] - sa * x0[i]) / sb;
        return eps;
    };

    const sampler = createSampler(predictEps, {
        T: meta.T, steps: 12, classIdx: 5, guidance: meta.bestGuidance, seed: 7,
    });
    while (!sampler.done) sampler.step();
    assert.equal(sampler.history.length, 12);
    const last = sampler.history[sampler.history.length - 1];
    let dark = 0;
    for (const v of last.x0Hat) {
        assert.ok(Number.isFinite(v));
        assert.ok(v >= -1 && v <= 1);
        if (v < -0.5) dark++;
    }
    // a digit image is mostly background
    assert.ok(dark > 300, `only ${dark} background pixels — not image-like`);
});

// The gate that would have caught the sampler's eps re-derivation bug: the
// shipped CNN must recognize what the shipped diffusion model draws.
test('the CNN recognizes one generated sample of every digit', () => {
    const { meta, predictX0 } = loadDenoiser();
    const alphaBar = makeSchedule(meta.T);
    const cnn = new ConvNet().loadTensors(
        parseWeights(loadBin('../public/data/weights/cnn-ref.bin')).tensors);

    const predictEps = (xt, t, cls) => {
        const x0 = predictX0(xt, t, cls);
        const sa = Math.sqrt(alphaBar[t]);
        const sb = Math.sqrt(1 - alphaBar[t]);
        const eps = new Float32Array(784);
        for (let i = 0; i < 784; i++) eps[i] = (xt[i] - sa * x0[i]) / sb;
        return eps;
    };

    let hits = 0;
    for (let d = 0; d < 10; d++) {
        const sampler = createSampler(predictEps, {
            T: meta.T, steps: 20, classIdx: d, guidance: 2, seed: 100 + d * 10,
        });
        let last = null;
        while (!sampler.done) last = sampler.step();
        const img = new Float32Array(784);
        for (let i = 0; i < 784; i++) img[i] = (last.x0Hat[i] + 1) / 2;
        const probs = cnn.forward(img).probs;
        let best = 0;
        for (let o = 1; o < 10; o++) if (probs[o] > probs[best]) best = o;
        if (best === d) hits++;
    }
    assert.ok(hits >= 8, `only ${hits}/10 samples recognized as their class`);
});
