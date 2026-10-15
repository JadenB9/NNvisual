import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync, gzipSync } from 'node:zlib';
import { unpackDataset } from '../public/js/engine/mnist.js';

const toBuffer = (bytes) => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

// The site fetches the gzipped copies (a fifth of the bytes) and inflates them
// in the browser. They are made from the .bin files with `npm run data:gzip`,
// and this pins them to each other so the two can never drift.
for (const name of ['mnist-train', 'mnist-test']) {
    test(`${name}.bin.gz inflates to exactly ${name}.bin`, async () => {
        const raw = readFileSync(new URL(`../public/data/${name}.bin`, import.meta.url));
        const gz = readFileSync(new URL(`../public/data/${name}.bin.gz`, import.meta.url));
        assert.ok(gz.length < raw.length / 3, 'the gzip is worth shipping');
        assert.deepEqual(gunzipSync(gz), raw);
        const ds = await unpackDataset(toBuffer(gz));
        assert.equal(ds.count, raw.readUInt32LE(4));
    });
}

// unpackDataset sniffs the gzip magic, so a server that already inflated the
// file (or a plain .bin) parses the same.
test('unpackDataset accepts gzipped and plain bytes alike', async () => {
    const raw = readFileSync(new URL('./fixtures/mnist-200.bin', import.meta.url));
    const a = await unpackDataset(toBuffer(raw));
    assert.equal(a.count, 200);
    const b = await unpackDataset(toBuffer(gzipSync(raw)));
    assert.equal(b.count, 200);
    assert.deepEqual([...b.labels.slice(0, 10)], [...a.labels.slice(0, 10)]);
    assert.deepEqual([...b.pixels.slice(0, 784)], [...a.pixels.slice(0, 784)]);
});
