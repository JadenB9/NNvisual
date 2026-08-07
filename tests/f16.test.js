import { test } from 'node:test';
import assert from 'node:assert/strict';
import { f16ToF32, f32ToF16, encodeF16, decodeF16 } from '../public/js/engine/f16.js';

test('exactly representable values round-trip', () => {
    for (const v of [0, 1, -1, 0.5, -2.25, 1024, 0.09375, 65504]) {
        assert.equal(f16ToF32(f32ToF16(v)), v, String(v));
    }
});

test('specials', () => {
    assert.equal(f16ToF32(f32ToF16(Infinity)), Infinity);
    assert.equal(f16ToF32(f32ToF16(-Infinity)), -Infinity);
    assert.ok(Number.isNaN(f16ToF32(f32ToF16(NaN))));
    assert.equal(f16ToF32(f32ToF16(1e10)), Infinity); // overflow
    assert.equal(f16ToF32(f32ToF16(1e-10)), 0);       // underflow
    assert.ok(Object.is(f16ToF32(f32ToF16(-0)) * 1, -0));
});

test('subnormals survive', () => {
    const tiny = 2 ** -24; // smallest positive half
    assert.equal(f16ToF32(f32ToF16(tiny)), tiny);
    const sub = 3 * 2 ** -24;
    assert.equal(f16ToF32(f32ToF16(sub)), sub);
});

test('arbitrary values round-trip within half precision', () => {
    let x = 0.1;
    for (let i = 0; i < 200; i++) {
        x = (x * 7919.7) % 10 - 5;
        const back = f16ToF32(f32ToF16(x));
        const tol = Math.max(Math.abs(x) * 2 ** -10, 2 ** -24);
        assert.ok(Math.abs(back - x) <= tol, `${x} -> ${back}`);
    }
});

test('bulk encode/decode agree with scalar path', () => {
    const src = Float32Array.from([0.25, -3.5, 100.0625, 0.0001]);
    const enc = encodeF16(src);
    const dec = decodeF16(enc);
    for (let i = 0; i < src.length; i++) {
        assert.equal(dec[i], f16ToF32(f32ToF16(src[i])));
    }
});
