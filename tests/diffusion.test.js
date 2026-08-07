import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, fillGaussian } from '../public/js/engine/rng.js';
import { makeSchedule, forwardNoise, tEmbed, createSampler } from '../public/js/engine/diffusion.js';

test('cosine schedule runs from 1 toward 0, monotonically', () => {
    const abar = makeSchedule(200);
    assert.ok(Math.abs(abar[0] - 1) < 1e-6);
    assert.ok(abar[200] < 0.01);
    for (let t = 1; t <= 200; t++) assert.ok(abar[t] <= abar[t - 1] + 1e-9);
});

test('forward noising interpolates between image and noise', () => {
    const x0 = new Float32Array(784).fill(0.8);
    const eps = new Float32Array(784).fill(-1);
    const nearStart = forwardNoise(x0, eps, 0.999);
    const nearEnd = forwardNoise(x0, eps, 0.001);
    assert.ok(Math.abs(nearStart[0] - 0.8) < 0.05);
    assert.ok(Math.abs(nearEnd[0] - -1) < 0.05);
});

test('tEmbed layout: sin block then cos block, t=0 gives 0s and 1s', () => {
    const e = tEmbed(0, 64);
    for (let i = 0; i < 32; i++) {
        assert.equal(e[i], 0);
        assert.equal(e[32 + i], 1);
    }
    const e2 = tEmbed(37, 64);
    for (const v of e2) assert.ok(v >= -1 && v <= 1);
});

// An oracle model that knows the true noise lets DDIM recover x0 exactly:
// with eps-hat == eps, x0-hat = (x_t - sqrt(1-abar)*eps)/sqrt(abar) == x0.
test('DDIM with an oracle denoiser recovers the clean image', () => {
    const T = 200;
    const abar = makeSchedule(T);
    const rand = mulberry32(3);
    const x0 = new Float32Array(784);
    for (let i = 0; i < 784; i++) x0[i] = Math.max(-1, Math.min(1, (rand() * 2 - 1) * 0.9));
    const eps = fillGaussian(new Float32Array(784), mulberry32(4));

    const t = 140;
    const xt = forwardNoise(x0, eps, abar[t]);
    const sa = Math.sqrt(abar[t]);
    const sb = Math.sqrt(1 - abar[t]);
    for (let i = 0; i < 784; i++) {
        const rec = (xt[i] - sb * eps[i]) / sa;
        assert.ok(Math.abs(rec - x0[i]) < 1e-3, `at ${i}: ${rec} vs ${x0[i]}`);
    }
});

test('sampler emits every step and is seed-deterministic', () => {
    const calls = [];
    const fakePredict = (xt, t, cls) => {
        calls.push([t, cls]);
        const eps = new Float32Array(784);
        for (let i = 0; i < 784; i++) eps[i] = xt[i] * 0.5;
        return eps;
    };
    const s1 = createSampler(fakePredict, { steps: 10, classIdx: 3, guidance: 0, seed: 42 });
    while (!s1.done) s1.step();
    assert.equal(s1.history.length, 10);
    for (const rec of s1.history) {
        assert.equal(rec.xt.length, 784);
        assert.equal(rec.x0Hat.length, 784);
        for (const v of rec.x0Hat) assert.ok(v >= -1 && v <= 1);
    }
    // guidance 0 -> one model call per step, conditional only
    assert.equal(calls.length, 10);
    assert.ok(calls.every(([, c]) => c === 3));

    const s2 = createSampler(fakePredict, { steps: 10, classIdx: 3, guidance: 0, seed: 42 });
    while (!s2.done) s2.step();
    assert.deepEqual([...s1.history[9].xt], [...s2.history[9].xt]);
});

test('classifier-free guidance calls the null class too', () => {
    let nullCalls = 0;
    const fakePredict = (xt, t, cls) => {
        if (cls === 10) nullCalls++;
        return new Float32Array(784);
    };
    const s = createSampler(fakePredict, { steps: 5, classIdx: 7, guidance: 2, seed: 1 });
    while (!s.done) s.step();
    assert.equal(nullCalls, 5);
});
