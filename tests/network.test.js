import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../public/js/nn/rng.js';
import { xor } from '../public/js/nn/data.js';
import { Network, crossEntropy } from '../public/js/nn/network.js';

test('layer shapes match the size spec', () => {
    const net = new Network([2, 5, 3, 1], 'tanh', mulberry32(1));
    assert.equal(net.weights.length, 3);
    assert.equal(net.weights[0].length, 5);
    assert.equal(net.weights[0][0].length, 2);
    assert.equal(net.weights[1].length, 3);
    assert.equal(net.weights[1][0].length, 5);
    assert.equal(net.weights[2].length, 1);
    assert.equal(net.weights[2][0].length, 3);
});

test('output is a probability', () => {
    const rand = mulberry32(2);
    const net = new Network([2, 4, 1], 'tanh', mulberry32(3));
    for (let i = 0; i < 20; i++) {
        const p = net.predict(rand() * 2 - 1, rand() * 2 - 1);
        assert.ok(p > 0 && p < 1, `not a probability: ${p}`);
    }
});

test('seeded construction is deterministic', () => {
    const a = new Network([2, 4, 4, 1], 'relu', mulberry32(42));
    const b = new Network([2, 4, 4, 1], 'relu', mulberry32(42));
    assert.deepEqual(a.weights, b.weights);
    assert.deepEqual(a.biases, b.biases);
});

test('cross-entropy is finite even at saturated predictions', () => {
    assert.ok(Number.isFinite(crossEntropy(0, 1)));
    assert.ok(Number.isFinite(crossEntropy(1, 0)));
    assert.ok(crossEntropy(0.5, 1) > 0);
});

// The strongest correctness check in the suite: backprop's gradient of the
// mean batch loss must agree with central-difference numerical gradients.
// trainBatch with lr = 1 moves each weight by exactly minus its gradient,
// so the gradient can be read off a stepped clone without exposing any
// internals.
test('backprop matches numerical gradients', () => {
    const sizes = [2, 3, 1];
    for (const activation of ['tanh', 'sigmoid', 'relu']) {
        const net = new Network(sizes, activation, mulberry32(7));
        const rand = mulberry32(8);
        const batch = Array.from({ length: 5 }, () => ({
            x: rand() * 2 - 1,
            y: rand() * 2 - 1,
            label: rand() < 0.5 ? 0 : 1,
        }));

        const stepped = new Network(sizes, activation, mulberry32(7));
        stepped.trainBatch(batch, 1);

        const h = 1e-5;
        for (let l = 0; l < net.weights.length; l++) {
            for (let j = 0; j < net.weights[l].length; j++) {
                for (let i = 0; i < net.weights[l][j].length; i++) {
                    const analytic = net.weights[l][j][i] - stepped.weights[l][j][i];
                    const saved = net.weights[l][j][i];
                    net.weights[l][j][i] = saved + h;
                    const up = net.loss(batch);
                    net.weights[l][j][i] = saved - h;
                    const down = net.loss(batch);
                    net.weights[l][j][i] = saved;
                    const numeric = (up - down) / (2 * h);
                    assert.ok(
                        Math.abs(numeric - analytic) < 1e-4,
                        `${activation} w[${l}][${j}][${i}]: numeric ${numeric} vs analytic ${analytic}`
                    );
                }
                const analytic = net.biases[l][j] - stepped.biases[l][j];
                const saved = net.biases[l][j];
                net.biases[l][j] = saved + h;
                const up = net.loss(batch);
                net.biases[l][j] = saved - h;
                const down = net.loss(batch);
                net.biases[l][j] = saved;
                const numeric = (up - down) / (2 * h);
                assert.ok(
                    Math.abs(numeric - analytic) < 1e-4,
                    `${activation} b[${l}][${j}]: numeric ${numeric} vs analytic ${analytic}`
                );
            }
        }
    }
});

test('training learns xor', () => {
    const data = xor(200, 0.05, mulberry32(13));
    const net = new Network([2, 8, 1], 'tanh', mulberry32(14));
    const before = net.loss(data);
    for (let epoch = 0; epoch < 600; epoch++) net.trainBatch(data, 0.8);
    const after = net.loss(data);
    assert.ok(after < before * 0.5, `loss barely moved: ${before} -> ${after}`);
    assert.ok(net.accuracy(data) >= 0.9, `accuracy too low: ${net.accuracy(data)}`);
});
