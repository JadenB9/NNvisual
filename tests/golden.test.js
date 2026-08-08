// Golden-fixture gates: the shipped weight files, loaded by the real JS
// engine, must reproduce the numpy trainers' outputs and hit accuracy
// thresholds on a held-out fixture set. If any of this drifts, the site is
// showing numbers the models never produced — these tests are the contract.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { parseWeights } from '../public/js/engine/weights.js';
import { parseDataset } from '../public/js/engine/mnist.js';
import { MLP } from '../public/js/engine/mlp.js';
import { ConvNet } from '../public/js/engine/conv.js';
import { Transformer } from '../public/js/engine/transformer.js';
import { mulberry32 } from '../public/js/engine/rng.js';

const at = (p) => fileURLToPath(new URL(p, import.meta.url));
const loadBin = (p) => {
    const b = readFileSync(at(p));
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const loadJson = (p) => JSON.parse(readFileSync(at(p), 'utf8'));

const fixtures = parseDataset(loadBin('fixtures/mnist-200.bin'));

test('shipped MNIST subsets parse and are stratified', () => {
    const train = parseDataset(loadBin('../public/data/mnist-train.bin'));
    assert.equal(train.count, 6000);
    const counts = new Array(10).fill(0);
    for (let i = 0; i < train.count; i++) counts[train.labels[i]]++;
    for (let d = 0; d < 10; d++) assert.equal(counts[d], 600, `class ${d}`);

    const testSet = parseDataset(loadBin('../public/data/mnist-test.bin'));
    assert.equal(testSet.count, 1000);
    assert.equal(fixtures.count, 200);
});

test('reference MLP matches its golden forward and clears 90% on the fixture set', () => {
    const w = parseWeights(loadBin('../public/data/weights/mlp-ref.bin'));
    const net = new MLP(w.meta.sizes, { hidden: w.meta.hidden }, mulberry32(1)).loadTensors(w.tensors);

    const golden = loadJson('fixtures/mlp-golden.json');
    const probs = net.predict(Float32Array.from(golden.image));
    for (let i = 0; i < 10; i++) {
        assert.ok(Math.abs(probs[i] - golden.probs[i]) < 2e-3, `prob ${i}: ${probs[i]} vs ${golden.probs[i]}`);
    }

    let hits = 0;
    const x = new Float32Array(784);
    for (let i = 0; i < fixtures.count; i++) {
        const p = net.predict(fixtures.image(i, x));
        let best = 0;
        for (let o = 1; o < 10; o++) if (p[o] > p[best]) best = o;
        if (best === fixtures.labels[i]) hits++;
    }
    const acc = hits / fixtures.count;
    assert.ok(acc >= 0.9, `fixture accuracy ${acc}`);
});

test('reference CNN matches goldens (probs and flatten stage) and clears 95%', () => {
    const w = parseWeights(loadBin('../public/data/weights/cnn-ref.bin'));
    const net = new ConvNet().loadTensors(w.tensors);

    const golden = loadJson('fixtures/cnn-golden.json');
    const stages = net.forward(fixtures.image(0));
    for (let i = 0; i < 16; i++) {
        assert.ok(Math.abs(stages.flat[i] - golden.flatHead[i]) < 1e-2,
            `flat ${i}: ${stages.flat[i]} vs ${golden.flatHead[i]}`);
    }
    for (let i = 0; i < 10; i++) {
        assert.ok(Math.abs(stages.probs[i] - golden.probs[i]) < 2e-3, `prob ${i}`);
    }
    assert.equal(golden.label, fixtures.labels[0]);

    let hits = 0;
    const x = new Float32Array(784);
    for (let i = 0; i < fixtures.count; i++) {
        const p = net.forward(fixtures.image(i, x)).probs;
        let best = 0;
        for (let o = 1; o < 10; o++) if (p[o] > p[best]) best = o;
        if (best === fixtures.labels[i]) hits++;
    }
    const acc = hits / fixtures.count;
    assert.ok(acc >= 0.95, `fixture accuracy ${acc}`);
});

test('autoencoder reproduces its golden latent and reconstruction', () => {
    const w = parseWeights(loadBin('../public/data/weights/ae-ref.bin'));
    const enc = new MLP(w.meta.encSizes, { task: 'regress' }, mulberry32(1)).loadTensors(w.tensors, 'enc_');
    const dec = new MLP(w.meta.decSizes, { task: 'regress' }, mulberry32(2)).loadTensors(w.tensors, 'dec_');

    const golden = loadJson('fixtures/ae-golden.json');
    const z = enc.predict(fixtures.image(0));
    assert.ok(Math.abs(z[0] - golden.latent[0]) < 5e-3, `z0 ${z[0]} vs ${golden.latent[0]}`);
    assert.ok(Math.abs(z[1] - golden.latent[1]) < 5e-3, `z1 ${z[1]} vs ${golden.latent[1]}`);

    const xhat = dec.predict(Float32Array.from(golden.latent));
    for (let i = 0; i < 16; i++) {
        assert.ok(Math.abs(xhat[i] - golden.reconHead[i]) < 5e-3, `recon ${i}`);
    }
});

test('transformer reproduces its golden logits and yields a sane distribution', () => {
    const w = parseWeights(loadBin('../public/data/weights/transformer-ref.bin'));
    const model = new Transformer(w);
    const golden = loadJson('fixtures/transformer-golden.json');

    const fwd = model.forward(golden.tokens);
    const T = golden.tokens.length;
    const V = model.vocab;
    for (let i = 0; i < V; i++) {
        const got = fwd.logits[(T - 1) * V + i];
        assert.ok(Math.abs(got - golden.lastLogits[i]) < 5e-2,
            `logit ${i}: ${got} vs ${golden.lastLogits[i]}`);
    }

    const { probs } = model.nextDistribution(model.encode('ROMEO:'), 0.8, 20);
    let sum = 0;
    for (const p of probs) sum += p;
    assert.ok(Math.abs(sum - 1) < 1e-4);
});
