// Draws the network as a layered graph. Edge color carries the weight's
// sign (green positive, red negative); width and opacity carry magnitude,
// normalized against the current largest weight so the picture stays
// readable as weights grow during training.
const POS = '#6a9955';
const NEG = '#c25b4e';

function nodeY(count, idx, h) {
    const spacing = count > 1 ? Math.min(52, (h - 56) / (count - 1)) : 0;
    return h / 2 + (idx - (count - 1) / 2) * spacing;
}

export function drawDiagram(net, canvas) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const sizes = net.sizes;
    const padX = 46;
    const xs = sizes.map((_, l) => padX + (l * (w - padX * 2)) / (sizes.length - 1));

    let max = 1e-9;
    for (const layer of net.weights) {
        for (const row of layer) {
            for (const v of row) max = Math.max(max, Math.abs(v));
        }
    }

    for (let l = 0; l < net.weights.length; l++) {
        for (let j = 0; j < net.weights[l].length; j++) {
            for (let i = 0; i < net.weights[l][j].length; i++) {
                const v = net.weights[l][j][i];
                const m = Math.abs(v) / max;
                ctx.strokeStyle = v >= 0 ? POS : NEG;
                ctx.globalAlpha = 0.12 + 0.78 * m;
                ctx.lineWidth = 0.5 + 2.5 * m;
                ctx.beginPath();
                ctx.moveTo(xs[l], nodeY(sizes[l], i, h));
                ctx.lineTo(xs[l + 1], nodeY(sizes[l + 1], j, h));
                ctx.stroke();
            }
        }
    }
    ctx.globalAlpha = 1;

    ctx.font = '11px "Plex Mono", ui-monospace, monospace';
    ctx.textBaseline = 'middle';
    const inputLabels = ['x', 'y'];
    for (let l = 0; l < sizes.length; l++) {
        for (let j = 0; j < sizes[l]; j++) {
            const cx = xs[l];
            const cy = nodeY(sizes[l], j, h);
            ctx.beginPath();
            ctx.arc(cx, cy, 9, 0, Math.PI * 2);
            ctx.fillStyle = '#1a1e21';
            ctx.fill();
            ctx.strokeStyle = '#333a40';
            ctx.lineWidth = 1.2;
            ctx.stroke();
            if (l === 0) {
                ctx.fillStyle = '#8f8b83';
                ctx.textAlign = 'right';
                ctx.fillText(inputLabels[j] || '', cx - 16, cy);
            } else if (l === sizes.length - 1) {
                ctx.fillStyle = '#8f8b83';
                ctx.textAlign = 'left';
                ctx.fillText('out', cx + 16, cy);
            }
        }
    }
}
