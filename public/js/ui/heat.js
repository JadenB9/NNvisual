// Small heatmap renderers. Grayscale maps [0,max] onto bg->bright ink;
// signed maps [-max,max] onto red->bg->green (the site's err/ok hues).
const BG = [14, 16, 17];
const INK = [238, 235, 227];
const POS = [106, 153, 85];
const NEG = [194, 91, 78];

let scratch = null;

function ensureScratch(w, h) {
    if (!scratch || scratch.width !== w || scratch.height !== h) {
        scratch = document.createElement('canvas');
        scratch.width = w;
        scratch.height = h;
    }
    return scratch;
}

export function drawHeat(canvas, data, w, h, { signed = false, max = null } = {}) {
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
