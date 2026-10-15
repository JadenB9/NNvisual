// Minimal auto-scaling line chart for loss/accuracy histories. Series colors
// are token names (see ui/theme.js) so the chart follows the theme.
import { tokens } from './theme.js';

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

    // a second series drawn in the quiet color, so two runs can be compared
    pin(label) {
        const live = this.series[0];
        if (!live.data.length) return false;
        this.series[1] = { color: 'faint', label, data: live.data.slice() };
        this.draw();
        return true;
    }

    unpin() {
        this.series.length = 1;
        this.draw();
    }

    draw() {
        const t = tokens();
        const ctx = this.canvas.getContext('2d');
        const w = this.canvas.width;
        const h = this.canvas.height;
        ctx.clearRect(0, 0, w, h);
        ctx.font = '19px "Plex Mono", ui-monospace, monospace';

        const n = Math.max(...this.series.map((s) => s.data.length));
        if (n < 1) {
            ctx.fillStyle = t.faint;
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
        const xAt = (i, len) => pad + (i / Math.max(1, len - 1)) * (w - 2 * pad);
        const yAt = (v) => h - pad - ((v - this.yMin) / (max - this.yMin)) * (h - 2 * pad);

        ctx.strokeStyle = t.border;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(pad, h - pad);
        ctx.lineTo(w - pad, h - pad);
        ctx.stroke();

        // the pinned run goes underneath the live one
        for (const s of [...this.series].reverse()) {
            const color = t[s.color] || s.color;
            if (s.data.length === 1) {
                // one epoch in: a single point, no line yet
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(xAt(0, 1), yAt(s.data[0]), 5, 0, Math.PI * 2);
                ctx.fill();
                continue;
            }
            if (s.data.length < 2) continue;
            const step = Math.max(1, Math.floor(s.data.length / 240));
            ctx.strokeStyle = color;
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

        ctx.fillStyle = t.faint;
        ctx.textAlign = 'left';
        ctx.fillText(max.toFixed(2), 4, 20);
        let lx = pad;
        for (const s of this.series) {
            const last = s.data[s.data.length - 1];
            if (last == null) continue;
            ctx.fillStyle = t[s.color] || s.color;
            ctx.fillText(`${s.label} ${last.toFixed(3)}`, lx, h - 6);
            lx += ctx.measureText(`${s.label} ${last.toFixed(3)}`).width + 22;
        }
    }
}
