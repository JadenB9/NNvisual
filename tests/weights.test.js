import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseWeights } from '../public/js/engine/weights.js';
import { encodeF16 } from '../public/js/engine/f16.js';

// Build an NNW1 buffer by hand, the way training/export.py does.
function buildFile(meta, tensorSpecs) {
    let offset = 0;
    const tensors = [];
    const payloads = [];
    for (const t of tensorSpecs) {
        const bytes = t.bytes;
        tensors.push({ name: t.name, shape: t.shape, dtype: t.dtype, offset, count: t.count });
        payloads.push(bytes);
        offset += bytes.byteLength;
        const pad = (4 - (offset % 4)) % 4;
        if (pad) {
            payloads.push(new Uint8Array(pad));
            offset += pad;
        }
    }
    const header = new TextEncoder().encode(JSON.stringify({ meta, tensors }));
    const total = 8 + header.length + offset;
    const buf = new ArrayBuffer(total);
    const u8 = new Uint8Array(buf);
    u8.set([78, 78, 87, 49], 0); // "NNW1"
    new DataView(buf).setUint32(4, header.length, true);
    u8.set(header, 8);
    let at = 8 + header.length;
    for (const p of payloads) {
        u8.set(new Uint8Array(p.buffer || p, p.byteOffset || 0, p.byteLength), at);
        at += p.byteLength;
    }
    return buf;
}

test('parses f32, f16, and u8 tensors with padding between them', () => {
    const f32 = Float32Array.from([1, 2, 3, -4]);
    const f16src = Float32Array.from([0.5, -1.25, 2]);
    const u8 = Uint8Array.from([9, 8, 7, 6, 5]);
    const buf = buildFile({ kind: 'test', vocab: 3 }, [
        { name: 'a', shape: [2, 2], dtype: 'f32', count: 4, bytes: f32 },
        { name: 'b', shape: [3], dtype: 'f16', count: 3, bytes: encodeF16(f16src) },
        { name: 'c', shape: [5], dtype: 'u8', count: 5, bytes: u8 },
    ]);

    const { meta, tensors } = parseWeights(buf);
    assert.equal(meta.kind, 'test');
    assert.deepEqual(tensors.a.shape, [2, 2]);
    assert.deepEqual([...tensors.a.data], [1, 2, 3, -4]);
    assert.deepEqual([...tensors.b.data], [0.5, -1.25, 2]);
    assert.deepEqual([...tensors.c.data], [9, 8, 7, 6, 5]);
});

test('rejects a wrong magic', () => {
    const buf = new ArrayBuffer(16);
    new Uint8Array(buf).set([88, 88, 88, 88]);
    assert.throws(() => parseWeights(buf), /magic/);
});
