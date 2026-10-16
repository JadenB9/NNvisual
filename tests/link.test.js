import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseLink, buildLink } from '../public/js/ui/link.js';
import { typingIn } from '../public/js/ui/keys.js';

const SPEC = {
    lr: { num: [0.01, 1] },
    batch: { oneOf: ['16', '32', '64'] },
    layers: { ints: [2, 48, 3] },
    seed: { int: [1, 99] },
    prompt: { text: 64 },
};

test('a good link round-trips', () => {
    const hash = buildLink({ lr: 0.05, batch: '64', layers: '32,16', seed: 7, prompt: 'ROMEO:' });
    assert.equal(hash, '#lr=0.05&batch=64&layers=32%2C16&seed=7&prompt=ROMEO%3A');
    assert.deepEqual(parseLink(hash, SPEC), { lr: 0.05, batch: '64', layers: [32, 16], seed: 7, prompt: 'ROMEO:' });
});

test('out-of-range and malformed values are dropped, not clamped', () => {
    const got = parseLink('#lr=9&batch=48&layers=1,16&seed=7.5&prompt=' + 'x'.repeat(65), SPEC);
    assert.deepEqual(got, {});
});

test('missing keys stay missing and extra keys are ignored', () => {
    assert.deepEqual(parseLink('#seed=3&evil=1', SPEC), { seed: 3 });
    assert.deepEqual(parseLink('', SPEC), {});
    assert.deepEqual(parseLink('#layers=8,8,8,8', SPEC), {});
    assert.deepEqual(parseLink('#layers=8,8,8', SPEC), { layers: [8, 8, 8] });
});

test('decimal lists parse and reject junk', () => {
    const spec = { kernel: { nums: [-9, 9, 9] } };
    assert.deepEqual(parseLink('#kernel=0,-1,0.11,4', spec), { kernel: [0, -1, 0.11, 4] });
    assert.deepEqual(parseLink('#kernel=0,x,1', spec), {});
    assert.deepEqual(parseLink('#kernel=10,0', spec), {});
    assert.deepEqual(parseLink('#kernel=1,1,1,1,1,1,1,1,1,1', spec), {});
});

test('shortcuts stay quiet while typing', () => {
    assert.equal(typingIn({ tagName: 'INPUT' }), true);
    assert.equal(typingIn({ tagName: 'TEXTAREA' }), true);
    assert.equal(typingIn({ tagName: 'SELECT' }), true);
    assert.equal(typingIn({ tagName: 'DIV', isContentEditable: true }), true);
    assert.equal(typingIn({ tagName: 'BUTTON' }), false);
    assert.equal(typingIn({ tagName: 'CANVAS' }), false);
    assert.equal(typingIn(null), false);
});
