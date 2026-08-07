import { decodeF16 } from './f16.js';

// Weight file format ("NNW1"):
//   bytes 0-3   magic "NNW1"
//   bytes 4-7   u32 LE header length H
//   bytes 8..   H bytes of UTF-8 JSON:
//                 { meta: {...}, tensors: [{ name, shape, dtype, offset, count }] }
//   then        payload; each tensor's offset is relative to the payload start
// dtype: "f32" | "f16" | "u8". Offsets are element-aligned by construction
// (the packer pads the payload so every tensor starts on a 4-byte boundary).

export function parseWeights(buffer) {
    const view = new DataView(buffer);
    const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (magic !== 'NNW1') throw new Error(`bad weights file: magic ${magic}`);
    const headerLen = view.getUint32(4, true);
    const headerBytes = new Uint8Array(buffer, 8, headerLen);
    const header = JSON.parse(new TextDecoder().decode(headerBytes));
    const payloadStart = 8 + headerLen;

    const tensors = {};
    for (const t of header.tensors) {
        const start = payloadStart + t.offset;
        let data;
        if (t.dtype === 'f32') {
            data = new Float32Array(buffer.slice(start, start + t.count * 4));
        } else if (t.dtype === 'f16') {
            data = decodeF16(new Uint16Array(buffer.slice(start, start + t.count * 2)));
        } else if (t.dtype === 'u8') {
            data = new Uint8Array(buffer.slice(start, start + t.count));
        } else {
            throw new Error(`unknown dtype ${t.dtype} for ${t.name}`);
        }
        tensors[t.name] = { shape: t.shape, data };
    }
    return { meta: header.meta || {}, tensors };
}

// Fetch with byte-level progress for the loading readouts.
export async function fetchWeights(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const total = Number(res.headers.get('content-length')) || 0;
    if (!res.body || !onProgress) {
        return parseWeights(await res.arrayBuffer());
    }
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        onProgress(received, total);
    }
    const all = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { all.set(c, off); off += c.length; }
    return parseWeights(all.buffer);
}
