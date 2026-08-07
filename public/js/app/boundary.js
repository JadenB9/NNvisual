// Renders the model's current belief across the whole input square by
// evaluating the net on a coarse grid and letting canvas smoothing upscale
// it, then scatters the training data on top.
const GRID = 56;
const EXTENT = 1.15; // world units shown; slightly beyond the data range

const BG = [14, 16, 17];      // --bg
const ONE = [106, 153, 85];   // --ok, class 1
const ZERO = [194, 91, 78];   // --err, class 0

let cell = null; // reusable offscreen canvas for the low-res field

export function drawBoundary(net, points, canvas) {
    const ctx = canvas.getContext('2d');
    const size = canvas.width;
    if (!cell) {
        cell = document.createElement('canvas');
        cell.width = GRID;
        cell.height = GRID;
    }
    const cctx = cell.getContext('2d');
    const img = cctx.createImageData(GRID, GRID);

    let k = 0;
    for (let gy = 0; gy < GRID; gy++) {
        const y = EXTENT - (gy / (GRID - 1)) * 2 * EXTENT; // canvas top = +y
        for (let gx = 0; gx < GRID; gx++) {
            const x = -EXTENT + (gx / (GRID - 1)) * 2 * EXTENT;
            const p = net.predict(x, y);
            const c = p >= 0.5 ? ONE : ZERO;
            const a = 0.1 + 0.5 * Math.abs(p - 0.5) * 2; // confidence -> opacity
            img.data[k++] = BG[0] + (c[0] - BG[0]) * a;
            img.data[k++] = BG[1] + (c[1] - BG[1]) * a;
            img.data[k++] = BG[2] + (c[2] - BG[2]) * a;
            img.data[k++] = 255;
        }
    }
    cctx.putImageData(img, 0, 0);

    ctx.clearRect(0, 0, size, size);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(cell, 0, 0, size, size);

    for (const p of points) {
        const px = ((p.x + EXTENT) / (2 * EXTENT)) * size;
        const py = ((EXTENT - p.y) / (2 * EXTENT)) * size;
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 1 ? 'rgb(106,153,85)' : 'rgb(194,91,78)';
        ctx.fill();
        ctx.strokeStyle = '#0e1011';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}
