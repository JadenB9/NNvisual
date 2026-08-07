import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../public/js/engine/rng.js';
import { MLP } from '../public/js/engine/mlp.js';

function randBatch(batch, dim, rand) {
    const x = new Float32Array(batch * dim);
    for (let i = 0; i < x.length; i++) x[i] = rand() * 2 - 1;
    return x;
}

test('forward shapes and probabilities', () => {
    const net = new MLP([4, 6, 3], {}, mulberry32(1));
    const rand = mulberry32(2);
    const x = randBatch(5, 4, rand);
    const { out } = net.forwardBatch(x, 5);
    assert.equal(out.length, 15);
    for (let b = 0; b < 5; b++) {
        let sum = 0;
        for (let o = 0; o < 3; o++) {
            sum += out[b * 3 + o];
            assert.ok(out[b * 3 + o] > 0 && out[b * 3 + o] < 1);
        }
        assert.ok(Math.abs(sum - 1) < 1e-5);
    }
});

test('seeded construction is deterministic', () => {
    const a = new MLP([3, 5, 2], {}, mulberry32(9));
    const b = new MLP([3, 5, 2], {}, mulberry32(9));
    assert.deepEqual([...a.weights[0]], [...b.weights[0]]);
    assert.deepEqual([...a.weights[1]], [...b.weights[1]]);
});

// SGD with lr=1 and momentum=0 moves each weight by exactly minus its
// gradient, so the gradient of the mean batch loss can be read off a stepped
// clone and compared to central differences of evalLoss.
function gradientCheck(task, hidden, labelsFor) {
    const sizes = [3, 4, task === 'classify' ? 3 : 2];
    const rand = mulberry32(7);
    const batch = 4;
    const x = randBatch(batch, 3, rand);
    const labels = labelsFor(batch, rand);

    const net = new MLP(sizes, { task, hidden }, mulberry32(11));
    const stepped = new MLP(sizes, { task, hidden }, mulberry32(11));
    stepped.trainBatch(x, labels, batch, 1, 0);

    const h = 1e-3;
    for (let l = 0; l < net.depth; l++) {
        for (let i = 0; i < net.weights[l].length; i++) {
            const analytic = net.weights[l][i] - stepped.weights[l][i];
            const saved = net.weights[l][i];
            net.weights[l][i] = saved + h;
            const up = net.evalLoss(x, labels, batch);
            net.weights[l][i] = saved - h;
            const down = net.evalLoss(x, labels, batch);
            net.weights[l][i] = saved;
            const numeric = (up - down) / (2 * h);
            assert.ok(
                Math.abs(numeric - analytic) < 2e-3,
                `${task}/${hidden} w[${l}][${i}]: numeric ${numeric} vs analytic ${analytic}`
            );
        }
        for (let i = 0; i < net.biases[l].length; i++) {
            const analytic = net.biases[l][i] - stepped.biases[l][i];
            const saved = net.biases[l][i];
            net.biases[l][i] = saved + h;
            const up = net.evalLoss(x, labels, batch);
            net.biases[l][i] = saved - h;
            const down = net.evalLoss(x, labels, batch);
            net.biases[l][i] = saved;
            const numeric = (up - down) / (2 * h);
            assert.ok(
                Math.abs(numeric - analytic) < 2e-3,
                `${task}/${hidden} b[${l}][${i}]: numeric ${numeric} vs analytic ${analytic}`
            );
        }
    }
}

test('backprop matches numerical gradients (classify, relu)', () => {
    gradientCheck('classify', 'relu', (batch, rand) => {
        const y = new Uint8Array(batch);
        for (let i = 0; i < batch; i++) y[i] = Math.floor(rand() * 3);
        return y;
    });
});

test('backprop matches numerical gradients (classify, tanh)', () => {
    gradientCheck('classify', 'tanh', (batch, rand) => {
        const y = new Uint8Array(batch);
        for (let i = 0; i < batch; i++) y[i] = Math.floor(rand() * 3);
        return y;
    });
});

test('backprop matches numerical gradients (regress)', () => {
    gradientCheck('regress', 'relu', (batch, rand) => {
        const y = new Float32Array(batch * 2);
        for (let i = 0; i < y.length; i++) y[i] = rand() * 2 - 1;
        return y;
    });
});

test('learns a separable 2-D problem', () => {
    const rand = mulberry32(21);
    const n = 200;
    const x = new Float32Array(n * 2);
    const y = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
        const cls = i % 2;
        x[i * 2] = (cls ? 0.6 : -0.6) + (rand() - 0.5) * 0.5;
        x[i * 2 + 1] = (cls ? -0.4 : 0.4) + (rand() - 0.5) * 0.5;
        y[i] = cls;
    }
    const net = new MLP([2, 8, 2], {}, mulberry32(22));
    for (let epoch = 0; epoch < 60; epoch++) net.trainBatch(x, y, n, 0.5);
    assert.ok(net.accuracy(x, y, n) >= 0.95, `acc ${net.accuracy(x, y, n)}`);
});

test('trainBatch exposes real gradients for the backprop viz', () => {
    const rand = mulberry32(30);
    const net = new MLP([3, 4, 2], {}, mulberry32(31));
    const x = randBatch(2, 3, rand);
    net.trainBatch(x, Uint8Array.from([0, 1]), 2, 0.1);
    assert.equal(net.lastGrads.gw.length, 2);
    assert.equal(net.lastGrads.gw[0].length, 12);
    let nonzero = 0;
    for (const g of net.lastGrads.gw[1]) if (g !== 0) nonzero++;
    assert.ok(nonzero > 0);
});

test('loadTensors installs pretrained weights', () => {
    const net = new MLP([2, 3, 2], {}, mulberry32(40));
    const tensors = {
        w0: { shape: [3, 2], data: Float32Array.from([1, 2, 3, 4, 5, 6]) },
        b0: { shape: [3], data: Float32Array.from([0.1, 0.2, 0.3]) },
        w1: { shape: [2, 3], data: Float32Array.from([1, 0, 0, 0, 1, 0]) },
        b1: { shape: [2], data: Float32Array.from([0, 0]) },
    };
    net.loadTensors(tensors);
    assert.deepEqual([...net.weights[0]], [1, 2, 3, 4, 5, 6]);
    assert.throws(() => net.loadTensors({ w0: tensors.w0 }), /missing/);
});
