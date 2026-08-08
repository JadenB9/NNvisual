import { drawHeat } from '../ui/heat.js';

// Renders the whole forward pass: input grid -> unrolled 784 strip -> layer
// columns -> output digits. Everything drawn is read straight off the last
// real forward pass (acts) or the last real backward pass (grads).
const POS = '#6a9955';
const NEG = '#c25b4e';
const INK = '#eeebe3';

export class Pipeline {
    constructor(canvas, inspector, inspHeat, inspText) {
        this.canvas = canvas;
        this.inspector = inspector;
        this.inspHeat = inspHeat;
        this.inspText = inspText;
        this.net = null;
        this.acts = null;
        this.input = null;
        this.pulse = 0;       // forward sweep progress 0..1 (1 = idle)
        this.bpPulse = 1;     // backward sweep progress
        this.hover = null;
        this.geom = null;

        canvas.addEventListener('pointermove', (e) => this.onMove(e));
        canvas.addEventListener('pointerleave', () => {
            this.hover = null;
            this.inspector.classList.remove('show');
            this.draw();
        });
    }

    setState(net, acts, input) {
        this.net = net;
        this.acts = acts;
        this.input = input;
    }

    startPulse() { this.pulse = 0; }
    startBackprop() { if (this.net && this.net.lastGrads) this.bpPulse = 0; }

    // node positions for the current net + canvas size
    layout() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;
        if (this.canvas.width !== Math.round(w * dpr)) this.canvas.width = Math.round(w * dpr);
        if (this.canvas.height !== Math.round(h * dpr)) this.canvas.height = Math.round(h * dpr);

        const sizes = this.net.sizes;
        const gridSize = Math.min(120, h * 0.3);
        const gridX = 12;
        const stripX = gridX + gridSize + 46;
        const outX = w - 46;
        const layerCount = sizes.length - 1; // hidden layers + output columns
        const span = outX - (stripX + 60);
        const xs = [stripX];
        for (let l = 1; l < sizes.length; l++) {
            xs.push(stripX + 60 + ((l - 1) / Math.max(1, layerCount - 1)) * span);
        }
        const nodeY = (count, idx) => {
            const usable = h - 70;
            const spacing = Math.min(usable / Math.max(count - 1, 1), 30);
            return h / 2 + (idx - (count - 1) / 2) * spacing;
        };
        this.geom = { w, h, dpr, gridX, gridSize, stripX, xs, nodeY, stripTop: 34, stripH: h - 68 };
        return this.geom;
    }

    nodeAt(px, py) {
        if (!this.geom || !this.net) return null;
        const { xs, nodeY } = this.geom;
        for (let l = 1; l < this.net.sizes.length; l++) {
            const count = this.net.sizes[l];
            for (let i = 0; i < count; i++) {
                const dx = px - xs[l];
                const dy = py - nodeY(count, i);
                if (dx * dx + dy * dy < 144) return { layer: l, index: i };
            }
        }
        return null;
    }

    onMove(e) {
        if (!this.net) return;
        const r = this.canvas.getBoundingClientRect();
        const hit = this.nodeAt(e.clientX - r.left, e.clientY - r.top);
        const changed = JSON.stringify(hit) !== JSON.stringify(this.hover);
        this.hover = hit;
        if (hit) {
            this.showInspector(hit, e.clientX - r.left, e.clientY - r.top);
        } else {
            this.inspector.classList.remove('show');
        }
        if (changed) this.draw();
    }

    showInspector(hit, px, py) {
        const net = this.net;
        const l = hit.layer - 1; // weight matrix index feeding this node
        const idx = hit.index;
        const nIn = net.sizes[l];
        const w = net.weights[l].subarray(idx * nIn, (idx + 1) * nIn);
        const z = this.acts ? this.acts.zs?.[l]?.[idx] : null;
        const a = this.acts ? this.acts.acts[hit.layer][idx] : null;

        if (nIn === 784) {
            this.inspHeat.hidden = false;
            drawHeat(this.inspHeat, w, 28, 28, { signed: true });
        } else {
            this.inspHeat.hidden = false;
            drawHeat(this.inspHeat, w, nIn, 1, { signed: true });
        }
        const name = hit.layer === net.sizes.length - 1 ? `output "${idx}"` : `hidden${hit.layer} #${idx}`;
        const lines = [
            name,
            `bias    ${net.biases[l][idx].toFixed(3)}`,
            z != null ? `z (sum) ${z.toFixed(3)}` : '',
            a != null ? `act     ${a.toFixed(3)}` : '',
            nIn === 784 ? 'weights drawn as 28×28 —' : `${nIn} incoming weights above`,
            nIn === 784 ? 'this is what it looks for' : '',
        ].filter(Boolean);
        this.inspText.textContent = lines.join('\n');

        const wrap = this.canvas.parentElement.getBoundingClientRect();
        const bx = Math.min(px + 18, wrap.width - 250);
        const by = Math.min(py + 12, wrap.height - 190);
        this.inspector.style.left = `${bx}px`;
        this.inspector.style.top = `${by}px`;
        this.inspector.classList.add('show');
    }

    draw() {
        if (!this.net) return;
        const g = this.layout();
        const ctx = this.canvas.getContext('2d');
        ctx.setTransform(g.dpr, 0, 0, g.dpr, 0, 0);
        ctx.clearRect(0, 0, g.w, g.h);

        const net = this.net;
        const acts = this.acts;
        const sizes = net.sizes;
        const grads = net.lastGrads;
        const bpActive = this.bpPulse < 1;

        // ---- input grid + unrolled strip ----
        if (this.input) {
            this.drawInputGrid(ctx, g);
            this.drawStrip(ctx, g);
        }

        // ---- edges ----
        // strip -> first layer: strongest |w * x| contributions per neuron
        const inActs = acts ? acts.acts[0] : null;
        const firstW = net.weights[0];
        const n1 = sizes[1];
        for (let j = 0; j < n1; j++) {
            const picks = topContrib(firstW, inActs, j, 784, 5);
            for (const p of picks) {
                const sy = g.stripTop + (p.i / 783) * g.stripH;
                this.edge(ctx, g.stripX + 8, sy, g.xs[1] - 9, g.nodeY(n1, j), p.w, p.m, 1, bpActive, grads, 0, j, p.i);
            }
        }
        // later layers: all edges
        for (let l = 1; l < sizes.length - 1; l++) {
            const wArr = net.weights[l];
            const from = sizes[l];
            const to = sizes[l + 1];
            const src = acts ? acts.acts[l] : null;
            let max = 1e-9;
            for (let j = 0; j < to; j++) {
                for (let i = 0; i < from; i++) {
                    const m = Math.abs(wArr[j * from + i] * (src ? src[i] : 1));
                    if (m > max) max = m;
                }
            }
            for (let j = 0; j < to; j++) {
                for (let i = 0; i < from; i++) {
                    const wv = wArr[j * from + i];
                    const m = Math.abs(wv * (src ? src[i] : 1)) / max;
                    if (m < 0.04) continue;
                    this.edge(ctx, g.xs[l] + 9, g.nodeY(from, i), g.xs[l + 1] - 9, g.nodeY(to, j), wv, m, l + 1, bpActive, grads, l, j, i);
                }
            }
        }

        // forward pulse band
        if (this.pulse < 1) {
            const x = g.stripX + this.pulse * (g.xs[sizes.length - 1] - g.stripX);
            const grad = ctx.createLinearGradient(x - 30, 0, x + 30, 0);
            grad.addColorStop(0, 'rgba(238,235,227,0)');
            grad.addColorStop(0.5, 'rgba(238,235,227,0.10)');
            grad.addColorStop(1, 'rgba(238,235,227,0)');
            ctx.fillStyle = grad;
            ctx.fillRect(x - 30, 20, 60, g.h - 40);
        }

        // ---- nodes ----
        ctx.font = '12px "Plex Mono", ui-monospace, monospace';
        for (let l = 1; l < sizes.length; l++) {
            const count = sizes[l];
            const vals = acts ? acts.acts[l] : null;
            let peak = 1e-9;
            if (vals) for (let i = 0; i < count; i++) peak = Math.max(peak, Math.abs(vals[i]));
            for (let i = 0; i < count; i++) {
                const x = g.xs[l];
                const y = g.nodeY(count, i);
                const v = vals ? Math.abs(vals[i]) / peak : 0;
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(238,235,227,${0.06 + v * 0.85})`;
                ctx.fill();
                ctx.strokeStyle = this.hover && this.hover.layer === l && this.hover.index === i
                    ? INK : '#333a40';
                ctx.lineWidth = this.hover && this.hover.layer === l && this.hover.index === i ? 2 : 1.2;
                ctx.stroke();
                if (l === sizes.length - 1) {
                    ctx.fillStyle = '#8f8b83';
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(String(i), x + 14, y);
                }
            }
        }

        // column captions
        ctx.fillStyle = '#5f5c55';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText('input 28×28', g.gridX + g.gridSize / 2, 24);
        ctx.fillText('784', g.stripX, 24);
        for (let l = 1; l < sizes.length - 1; l++) ctx.fillText(`hidden ${l} (${sizes[l]})`, g.xs[l], 24);
        ctx.fillText('output (10)', g.xs[sizes.length - 1], 24);
    }

    edge(ctx, x0, y0, x1, y1, w, m, stage, bpActive, grads, gl, gj, gi) {
        // during backprop replay recolor by the real gradient, sweeping
        // right-to-left; -grad is the direction the weight will move
        if (bpActive && grads) {
            const stages = this.net.sizes.length - 1;
            const stageStart = (stages - stage) / stages;
            const local = (this.bpPulse - stageStart) * stages;
            if (local <= 0) { this.plainEdge(ctx, x0, y0, x1, y1, w, m * 0.35); return; }
            const gv = grads.gw[gl][gj * this.net.sizes[gl] + gi];
            let gm = Math.abs(gv) / (this.gradPeak(gl) || 1e-9);
            gm = Math.min(1, gm);
            if (gm < 0.03) { this.plainEdge(ctx, x0, y0, x1, y1, w, m * 0.3); return; }
            ctx.strokeStyle = gv <= 0 ? POS : NEG; // weight moves opposite the gradient
            ctx.globalAlpha = 0.15 + 0.8 * gm * Math.min(1, local);
            ctx.lineWidth = 0.6 + 2.6 * gm;
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.stroke();
            ctx.globalAlpha = 1;
            return;
        }
        this.plainEdge(ctx, x0, y0, x1, y1, w, m);
    }

    plainEdge(ctx, x0, y0, x1, y1, w, m) {
        ctx.strokeStyle = w >= 0 ? POS : NEG;
        ctx.globalAlpha = 0.08 + 0.8 * Math.min(1, m);
        ctx.lineWidth = 0.5 + 2.2 * Math.min(1, m);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        ctx.globalAlpha = 1;
    }

    gradPeak(l) {
        if (this._gradPeakCache && this._gradPeakCache.grads === this.net.lastGrads) {
            return this._gradPeakCache.peaks[l];
        }
        const peaks = this.net.lastGrads.gw.map((gw) => {
            let p = 0;
            for (let i = 0; i < gw.length; i++) p = Math.max(p, Math.abs(gw[i]));
            return p;
        });
        this._gradPeakCache = { grads: this.net.lastGrads, peaks };
        return peaks[l];
    }

    drawInputGrid(ctx, g) {
        const cell = g.gridSize / 28;
        const top = g.h / 2 - g.gridSize / 2;
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const v = this.input[y * 28 + x];
                if (v < 0.02) continue;
                ctx.fillStyle = `rgba(238,235,227,${v})`;
                ctx.fillRect(g.gridX + x * cell, top + y * cell, cell + 0.5, cell + 0.5);
            }
        }
        ctx.strokeStyle = '#24292d';
        ctx.strokeRect(g.gridX, top, g.gridSize, g.gridSize);

        // unroll guides: corners of the grid flow to the strip's ends
        ctx.strokeStyle = 'rgba(95,92,85,0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(g.gridX + g.gridSize, top);
        ctx.lineTo(g.stripX - 3, g.stripTop);
        ctx.moveTo(g.gridX + g.gridSize, top + g.gridSize);
        ctx.lineTo(g.stripX - 3, g.stripTop + g.stripH);
        ctx.stroke();
    }

    drawStrip(ctx, g) {
        const n = 784;
        for (let i = 0; i < n; i++) {
            const v = this.input[i];
            if (v < 0.02) continue;
            const y = g.stripTop + (i / (n - 1)) * g.stripH;
            ctx.fillStyle = `rgba(238,235,227,${0.25 + v * 0.75})`;
            ctx.fillRect(g.stripX - 3, y - 0.6, 7, 1.4);
        }
        ctx.strokeStyle = '#333a40';
        ctx.strokeRect(g.stripX - 4.5, g.stripTop - 2, 10, g.stripH + 4);
    }

    tick() {
        let moving = false;
        if (this.pulse < 1) { this.pulse = Math.min(1, this.pulse + 0.06); moving = true; }
        if (this.bpPulse < 1) { this.bpPulse = Math.min(1, this.bpPulse + 0.02); moving = true; }
        return moving;
    }
}

function topContrib(w, x, j, nIn, k) {
    const picks = [];
    for (let i = 0; i < nIn; i++) {
        const xv = x ? x[i] : 0;
        if (xv < 0.05) continue;
        const wv = w[j * nIn + i];
        const m = Math.abs(wv * xv);
        if (picks.length < k) {
            picks.push({ i, w: wv, m });
            picks.sort((a, b) => a.m - b.m);
        } else if (m > picks[0].m) {
            picks[0] = { i, w: wv, m };
            picks.sort((a, b) => a.m - b.m);
        }
    }
    // normalize magnitudes for drawing
    const peak = picks.length ? picks[picks.length - 1].m : 1;
    return picks.map((p) => ({ ...p, m: p.m / (peak || 1) }));
}
