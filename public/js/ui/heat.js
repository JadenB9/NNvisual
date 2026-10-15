// Small heatmap renderers. Grayscale maps [0,max] onto bg->ink; signed maps
// [-max,max] onto err->bg->ok (the site's red/green). The colors are the page's
// theme tokens, read at draw time.
import { tokens, hexToRgb } from './theme.js';

let scratch = null;

function ensureScratch(w, h) {
    if (!scratch || scratch.width !== w || scratch.height !== h) {
        scratch = document.createElement('canvas');
        scratch.width = w;
        scratch.height = h;
    }
    return scratch;
}

// Every canvas remembers its last draw so a theme switch can repaint all of
// them at once (sample strips, galleries, film frames) without each page
// keeping its own list. Pruned when it grows, since pages rebuild strips.
const painted = new Map();

function remember(canvas, fn, args) {
    if (painted.size > 600) {
        for (const c of painted.keys()) if (!c.isConnected) painted.delete(c);
    }
    painted.set(canvas, { fn, args });
}

export function repaintHeatmaps() {
    for (const [canvas, { fn, args }] of [...painted]) {
        if (canvas.isConnected) fn(canvas, ...args);
        else painted.delete(canvas);
    }
}

function palette() {
    const t = tokens();
    return { BG: hexToRgb(t.bg), INK: hexToRgb(t.ink), POS: hexToRgb(t.ok), NEG: hexToRgb(t.err) };
}

export function drawHeat(canvas, data, w, h, { signed = false, max = null } = {}) {
    remember(canvas, drawHeat, [data, w, h, { signed, max }]);
    const { BG, INK, POS, NEG } = palette();
    let peak = max;
    if (peak == null) {
        peak = 1e-9;
        for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i]));
    }
    const off = ensureScratch(w, h);
    const octx = off.getContext('2d');
    const img = octx.createImageData(w, h);
    let k = 0;
    for (let i = 0; i < w * h; i++) {
        const v = data[i] / peak;
        let c;
        let a;
        if (signed) {
            c = v >= 0 ? POS : NEG;
            a = Math.min(1, Math.abs(v));
        } else {
            c = INK;
            a = Math.min(1, Math.max(0, v));
        }
        img.data[k++] = BG[0] + (c[0] - BG[0]) * a;
        img.data[k++] = BG[1] + (c[1] - BG[1]) * a;
        img.data[k++] = BG[2] + (c[2] - BG[2]) * a;
        img.data[k++] = 255;
    }
    octx.putImageData(img, 0, 0);
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
    return peak;
}

// [-1,1] images (diffusion space) onto grayscale.
export function drawSignedImage(canvas, data, w, h) {
    remember(canvas, drawSignedImage, [data, w, h]);
    const { BG, INK } = palette();
    const off = ensureScratch(w, h);
    const octx = off.getContext('2d');
    const img = octx.createImageData(w, h);
    let k = 0;
    for (let i = 0; i < w * h; i++) {
        const a = Math.min(1, Math.max(0, (data[i] + 1) / 2));
        img.data[k++] = BG[0] + (INK[0] - BG[0]) * a;
        img.data[k++] = BG[1] + (INK[1] - BG[1]) * a;
        img.data[k++] = BG[2] + (INK[2] - BG[2]) * a;
        img.data[k++] = 255;
    }
    octx.putImageData(img, 0, 0);
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

// forget a canvas that was cleared on purpose (nothing to repaint)
export function forgetHeat(canvas) {
    painted.delete(canvas);
}
