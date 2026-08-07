import { test } from 'node:test';
import assert from 'node:assert/strict';
import { activations } from '../public/js/nn/activations.js';

test('known values', () => {
    assert.equal(activations.tanh.f(0), 0);
    assert.equal(activations.sigmoid.f(0), 0.5);
    assert.equal(activations.relu.f(-2), 0);
    assert.equal(activations.relu.f(3), 3);
});

test('outputs stay in each activation range', () => {
    for (const x of [-4, -1, 0.5, 4]) {
        const t = activations.tanh.f(x);
        const s = activations.sigmoid.f(x);
        assert.ok(t >= -1 && t <= 1);
        assert.ok(s >= 0 && s <= 1);
        assert.ok(activations.relu.f(x) >= 0);
    }
});

// df is defined on the activation's *output*; check it against a numerical
// derivative of f. Points avoid relu's kink at 0, where the derivative is
// undefined anyway.
test('derivatives match numerical differentiation', () => {
    const h = 1e-6;
    for (const name of Object.keys(activations)) {
        const { f, df } = activations[name];
        for (const x of [-1.5, -0.3, 0.4, 1.2]) {
            const numeric = (f(x + h) - f(x - h)) / (2 * h);
            const analytic = df(f(x));
            assert.ok(
                Math.abs(numeric - analytic) < 1e-4,
                `${name} at ${x}: numeric ${numeric} vs analytic ${analytic}`
            );
        }
    }
});
