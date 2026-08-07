// Packed MNIST subsets ("NND1"):
//   bytes 0-3  magic "NND1"
//   bytes 4-7  u32 LE image count N
//   8..8+N     labels, one byte each
//   then       pixels, N * 784 bytes, row-major 28x28, 0..255
export function parseDataset(buffer) {
    const view = new DataView(buffer);
    const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (magic !== 'NND1') throw new Error(`bad dataset file: magic ${magic}`);
    const count = view.getUint32(4, true);
    const labels = new Uint8Array(buffer, 8, count);
    const pixels = new Uint8Array(buffer, 8 + count, count * 784);
    return {
        count,
        labels,
        pixels,
        // normalized [0,1] copy of image i
        image(i, out) {
            out = out || new Float32Array(784);
            const off = i * 784;
            for (let p = 0; p < 784; p++) out[p] = pixels[off + p] / 255;
            return out;
        },
        // fill a [batch,784] buffer from index list
        fillBatch(indices, start, batch, x, y) {
            for (let b = 0; b < batch; b++) {
                const i = indices[start + b];
                const off = i * 784;
                const xOff = b * 784;
                for (let p = 0; p < 784; p++) x[xOff + p] = pixels[off + p] / 255;
                y[b] = labels[i];
            }
        },
    };
}

export async function fetchDataset(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    if (!res.body || !onProgress) return parseDataset(await res.arrayBuffer());
    const total = Number(res.headers.get('content-length')) || 0;
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
    return parseDataset(all.buffer);
}

// Bilinear resample a [sh,sw] grayscale block to [dh,dw].
export function resampleBilinear(src, sh, sw, dh, dw) {
    const out = new Float32Array(dh * dw);
    const yScale = sh / dh;
    const xScale = sw / dw;
    for (let y = 0; y < dh; y++) {
        const sy = Math.min(sh - 1, Math.max(0, (y + 0.5) * yScale - 0.5));
        const y0 = Math.floor(sy);
        const y1 = Math.min(sh - 1, y0 + 1);
        const fy = sy - y0;
        for (let x = 0; x < dw; x++) {
            const sx = Math.min(sw - 1, Math.max(0, (x + 0.5) * xScale - 0.5));
            const x0 = Math.floor(sx);
            const x1 = Math.min(sw - 1, x0 + 1);
            const fx = sx - x0;
            const top = src[y0 * sw + x0] * (1 - fx) + src[y0 * sw + x1] * fx;
            const bot = src[y1 * sw + x0] * (1 - fx) + src[y1 * sw + x1] * fx;
            out[y * dw + x] = top * (1 - fy) + bot * fy;
        }
    }
    return out;
}

// Canonical MNIST normalization for user drawings, the reason drawn digits
// classify well: crop to the ink's bounding box, scale the longest side to
// 20px, then paste into 28x28 so the center of mass sits at the center.
export function preprocessDrawing(src) {
    let x0 = 28, y0 = 28, x1 = -1, y1 = -1;
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            if (src[y * 28 + x] > 0.05) {
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
            }
        }
    }
    if (x1 < 0) return new Float32Array(784); // blank canvas

    const cw = x1 - x0 + 1;
    const ch = y1 - y0 + 1;
    const crop = new Float32Array(cw * ch);
    for (let y = 0; y < ch; y++) {
        for (let x = 0; x < cw; x++) crop[y * cw + x] = src[(y0 + y) * 28 + (x0 + x)];
    }

    const scale = 20 / Math.max(cw, ch);
    const dw = Math.max(1, Math.round(cw * scale));
    const dh = Math.max(1, Math.round(ch * scale));
    const scaled = resampleBilinear(crop, ch, cw, dh, dw);

    // center of mass of the scaled ink
    let mass = 0, mx = 0, my = 0;
    for (let y = 0; y < dh; y++) {
        for (let x = 0; x < dw; x++) {
            const v = scaled[y * dw + x];
            mass += v;
            mx += v * x;
            my += v * y;
        }
    }
    const comX = mass > 0 ? mx / mass : (dw - 1) / 2;
    const comY = mass > 0 ? my / mass : (dh - 1) / 2;

    const out = new Float32Array(784);
    const offX = Math.min(28 - dw, Math.max(0, Math.round(13.5 - comX)));
    const offY = Math.min(28 - dh, Math.max(0, Math.round(13.5 - comY)));
    for (let y = 0; y < dh; y++) {
        for (let x = 0; x < dw; x++) {
            out[(offY + y) * 28 + (offX + x)] = scaled[y * dw + x];
        }
    }
    return out;
}
