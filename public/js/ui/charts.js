// Minimal auto-scaling line chart for loss/accuracy histories.
export class LineChart {
    constructor(canvas, series, { yMax = null, yMin = 0 } = {}) {
        this.canvas = canvas;
        this.series = series.map((s) => ({ ...s, data: [] }));
        this.yMax = yMax;
        this.yMin = yMin;
    }

    push(...values) {
        values.forEach((v, i) => {
            if (v != null && this.series[i]) this.series[i].data.push(v);
        });
        this.draw();
    }

    reset() {
        for (const s of this.series) s.data = [];
        this.draw();
    }

    draw() {
        const ctx = this.canvas.getContext('2d');
        const w = this.canvas.width;
        const h = this.canvas.height;
        ctx.clearRect(0, 0, w, h);
        ctx.font = '19px "Plex Mono", ui-monospace, monospace';

        const n = Math.max(...this.series.map((s) => s.data.length));
        if (n < 2) {
            ctx.fillStyle = '#5f5c55';
            ctx.textAlign = 'center';
            ctx.fillText('no data yet', w / 2, h / 2 + 6);
            return;
        }

        let max = this.yMax;
        if (max == null) {
            max = 1e-9;
            for (const s of this.series) for (const v of s.data) max = Math.max(max, v);
            max *= 1.05;
        }
        const pad = 26;
        const xAt = (i, len) => pad + (i / (len - 1)) * (w - 2 * pad);
        const yAt = (v) => h - pad - ((v - this.yMin) / (max - this.yMin)) * (h - 2 * pad);

        ctx.strokeStyle = '#24292d';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(pad, h - pad);
        ctx.lineTo(w - pad, h - pad);
        ctx.stroke();

        for (const s of this.series) {
            if (s.data.length < 2) continue;
            const step = Math.max(1, Math.floor(s.data.length / 240));
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            for (let i = 0; i < s.data.length; i += step) {
                const x = xAt(i, s.data.length);
                const y = yAt(s.data[i]);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            const last = s.data.length - 1;
            ctx.lineTo(xAt(last, s.data.length), yAt(s.data[last]));
            ctx.stroke();
        }

        ctx.fillStyle = '#5f5c55';
        ctx.textAlign = 'left';
        ctx.fillText(max.toFixed(2), 4, 20);
        let lx = pad;
        for (const s of this.series) {
            const last = s.data[s.data.length - 1];
            if (last == null) continue;
            ctx.fillStyle = s.color;
            ctx.fillText(`${s.label} ${last.toFixed(3)}`, lx, h - 6);
            lx += ctx.measureText(`${s.label} ${last.toFixed(3)}`).width + 22;
        }
    }
}
