// Training loss over epochs as a single auto-scaled line. Long runs are
// decimated so the polyline stays cheap no matter how far training goes.
export function drawLoss(history, canvas) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.font = '20px "Plex Mono", ui-monospace, monospace';
    ctx.fillStyle = '#5f5c55';

    if (history.length < 2) {
        ctx.textAlign = 'center';
        ctx.fillText('press train', w / 2, h / 2 + 7);
        return;
    }

    let max = 0;
    for (const v of history) max = Math.max(max, v);
    if (max === 0) max = 1;

    const pad = 14;
    const step = Math.max(1, Math.floor(history.length / 300));
    const xAt = (i) => pad + (i / (history.length - 1)) * (w - 2 * pad);
    const yAt = (v) => h - pad - (v / max) * (h - 2 * pad);

    ctx.strokeStyle = '#24292d';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    ctx.lineTo(w - pad, h - pad);
    ctx.stroke();

    ctx.strokeStyle = '#86c3d6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i = 0; i < history.length; i += step) {
        if (i === 0) ctx.moveTo(xAt(i), yAt(history[i]));
        else ctx.lineTo(xAt(i), yAt(history[i]));
    }
    const last = history.length - 1;
    ctx.lineTo(xAt(last), yAt(history[last]));
    ctx.stroke();

    ctx.textAlign = 'left';
    ctx.fillText(max.toFixed(2), pad, pad + 8);
    ctx.textAlign = 'right';
    ctx.fillText(history[last].toFixed(3), w - pad, pad + 8);
}
