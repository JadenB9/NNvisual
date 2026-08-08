import { fetchWeights } from '../engine/weights.js';
import { Transformer } from '../engine/transformer.js';
import { mulberry32 } from '../engine/rng.js';
import { drawHeat } from '../ui/heat.js';
import { Bars } from '../ui/bars.js';

const $ = (id) => document.getElementById(id);
const show = (ch) => (ch === '\n' ? '⏎' : ch === ' ' ? '␣' : ch);

const state = {
    model: null,
    tokens: [],        // full generated sequence (token ids)
    promptLen: 0,
    fwd: null,         // capture for the current context window
    ctxStart: 0,       // index of the first token in the window
    layer: 1,
    head: 0,
    selected: 0,       // selected position within the window
    pair: null,        // [row, col] in the matrix
    playing: false,
    speed: 4,
    lastStep: 0,
    temperature: 0.8,
    topK: 10,
    rand: mulberry32(Date.now() & 0xffffff),
    pca: null,
};

let nextBars = null;

// ---------- boot ----------
(async function boot() {
    const fill = $('load-fill');
    const text = $('load-text');
    try {
        const w = await fetchWeights('./data/weights/transformer-ref.bin', (got, total) => {
            const pct = total ? Math.round((got / total) * 100) : 0;
            fill.style.width = `${pct}%`;
            text.textContent = `fetching transformer… ${pct}%`;
        });
        state.model = new Transformer(w);
        state.valLoss = w.meta.valLoss;
    } catch (err) {
        text.textContent = `failed to load: ${err.message}`;
        throw err;
    }
    $('loader').classList.add('done');
    $('net-status').textContent =
        `char-gpt · ${state.model.vocab} vocab · ${state.model.nLayer}×${state.model.nHead} heads · val loss ${state.valLoss}`;

    nextBars = new Bars($('next-bars'), Array(10).fill(''));
    buildHeadPicker();
    computePca();
    buildEmbMap();
    buildCosExplorer();
    resetText();
    requestAnimationFrame(loop);
})();

// ---------- generation ----------
function resetText() {
    const prompt = $('prompt').value || 'ROMEO:';
    state.tokens = state.model.encode(prompt);
    if (state.tokens.length === 0) state.tokens = state.model.encode('ROMEO:');
    state.promptLen = state.tokens.length;
    state.playing = false;
    $('autoplay').textContent = 'generate';
    refresh(false);
}

function stepToken() {
    const m = state.model;
    const ctx = state.tokens.slice(-m.block);
    const { probs, fwd } = m.nextDistribution(ctx, state.temperature, state.topK);
    const tok = m.sample(probs, state.rand);
    state.tokens.push(tok);
    state.fwd = fwd;
    state.ctxStart = state.tokens.length - 1 - ctx.length + 0; // window start before append
    state.lastProbs = probs;
    refresh(true);
}

function refresh(fromStep) {
    const m = state.model;
    if (!fromStep) {
        const ctx = state.tokens.slice(-m.block);
        const { probs, fwd } = m.nextDistribution(ctx, state.temperature, state.topK);
        state.fwd = fwd;
        state.lastProbs = probs;
        state.ctxStart = state.tokens.length - ctx.length;
    }
    state.selected = state.fwd.T - 1;
    state.pair = null;
    renderText();
    renderChips();
    renderNextBars();
    renderAttn();
    renderTiles();
    renderPair();
}

function renderText() {
    const box = $('gen-text');
    const full = state.model.decode(state.tokens);
    box.textContent = '';
    if (state.tokens.length > state.promptLen) {
        // highlight only the freshly sampled character
        box.append(document.createTextNode(full.slice(0, -1)));
        const s = document.createElement('span');
        s.className = 'fresh';
        s.textContent = full.slice(-1);
        box.append(s);
    } else {
        box.append(document.createTextNode(full));
    }
    box.scrollTop = box.scrollHeight;
}

function renderChips() {
    const box = $('chips');
    box.textContent = '';
    const win = state.fwd.tokens;
    win.forEach((t, i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'chip-t'
            + (i === state.selected ? ' is-sel' : '')
            + (i === win.length - 1 && state.tokens.length > state.promptLen ? ' is-new' : '');
        b.textContent = show(state.model.chars[t]);
        b.title = `position ${i} · token ${t}`;
        b.addEventListener('click', () => {
            state.selected = i;
            renderChips();
            renderAttn();
            renderTiles();
        });
        box.append(b);
    });
    requestAnimationFrame(drawArcs);
}

function renderNextBars() {
    const probs = state.lastProbs;
    const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 10);
    nextBars.setLabels(order.map((i) => show(state.model.chars[i])));
    nextBars.update(order.map((i) => probs[i]), 0);
    const top = order[0];
    $('sample-note').innerHTML = state.tokens.length > state.promptLen
        ? `sampled <b>${show(state.model.decode([state.tokens[state.tokens.length - 1]]))}</b> at temperature ${state.temperature}`
        : `most likely next: <b>${show(state.model.chars[top])}</b> (${(probs[top] * 100).toFixed(1)}%)`;
}

// ---------- attention rendering ----------
function buildHeadPicker() {
    const box = $('head-pick');
    box.textContent = '';
    for (let l = 0; l < state.model.nLayer; l++) {
        for (let h = 0; h < state.model.nHead; h++) {
            const b = document.createElement('button');
            b.type = 'button';
            b.className = 'seg-btn' + (l === state.layer && h === state.head ? ' is-active' : '');
            b.textContent = `L${l + 1}·H${h + 1}`;
            b.addEventListener('click', () => {
                state.layer = l;
                state.head = h;
                box.querySelectorAll('.seg-btn').forEach((x) => x.classList.toggle('is-active', x === b));
                renderAttn();
                renderPair();
            });
            box.append(b);
        }
    }
}

function renderAttn() {
    if (!state.fwd) return;
    const T = state.fwd.T;
    const probs = state.fwd.layers[state.layer].heads[state.head].probs;
    const canvas = $('attn-matrix');
    const off = document.createElement('canvas');
    off.width = T;
    off.height = T;
    const octx = off.getContext('2d');
    const img = octx.createImageData(T, T);
    let k = 0;
    for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
            const v = Math.min(1, probs[i * T + j] * 2.2);
            img.data[k++] = 14 + (238 - 14) * v;
            img.data[k++] = 16 + (235 - 16) * v;
            img.data[k++] = 17 + (227 - 17) * v;
            img.data[k++] = 255;
        }
    }
    octx.putImageData(img, 0, 0);
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);

    // selected row guide
    const cell = canvas.width / T;
    ctx.strokeStyle = 'rgba(134,195,214,0.65)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, state.selected * cell, canvas.width, cell);
    if (state.pair) {
        ctx.strokeStyle = '#86c3d6';
        ctx.lineWidth = 2;
        ctx.strokeRect(state.pair[1] * cell, state.pair[0] * cell, cell, cell);
    }
    drawArcs();
}

function drawArcs() {
    const canvas = $('arcs');
    const chips = [...$('chips').children];
    if (!state.fwd || !chips.length) return;
    const wrap = $('chips').getBoundingClientRect();
    canvas.width = Math.max(200, Math.round(wrap.width));
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const T = state.fwd.T;
    const probs = state.fwd.layers[state.layer].heads[state.head].probs;
    const src = state.selected;
    const centers = chips.map((c) => c.offsetLeft + c.offsetWidth / 2);
    const y = canvas.height - 2;

    const entries = [];
    for (let j = 0; j <= src; j++) entries.push([j, probs[src * T + j]]);
    entries.sort((a, b) => b[1] - a[1]);
    for (const [j, p] of entries.slice(0, 8)) {
        if (p < 0.01 || j === src) continue;
        const x0 = centers[src];
        const x1 = centers[j];
        const lift = Math.min(56, 12 + Math.abs(x0 - x1) * 0.14);
        ctx.strokeStyle = '#86c3d6';
        ctx.globalAlpha = 0.25 + 0.7 * Math.min(1, p * 1.6);
        ctx.lineWidth = 0.8 + 5 * p;
        ctx.beginPath();
        ctx.moveTo(x0, y);
        ctx.quadraticCurveTo((x0 + x1) / 2, y - lift, x1, y);
        ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // self-attention dot
    ctx.fillStyle = '#86c3d6';
    ctx.globalAlpha = 0.3 + 0.7 * Math.min(1, probs[src * T + src] * 1.6);
    ctx.beginPath();
    ctx.arc(centers[src], y - 4, 3.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
}

$('attn-matrix').addEventListener('pointerdown', (e) => {
    const T = state.fwd.T;
    const r = e.currentTarget.getBoundingClientRect();
    const col = Math.floor(((e.clientX - r.left) / r.width) * T);
    const row = Math.floor(((e.clientY - r.top) / r.height) * T);
    if (col > row) return; // causal mask — the future is not an option
    state.pair = [row, col];
    state.selected = row;
    renderChips();
    renderAttn();
    renderTiles();
    renderPair();
});

function renderPair() {
    const box = $('pair-arith');
    if (!state.pair || !state.fwd) {
        box.textContent = 'click the matrix…';
        return;
    }
    const [row, col] = state.pair;
    const T = state.fwd.T;
    const layer = state.fwd.layers[state.layer];
    const raw = layer.heads[state.head].scores[row * T + col];
    const p = layer.heads[state.head].probs[row * T + col];
    const chars = state.model.chars;
    const rowCh = show(chars[state.fwd.tokens[row]]);
    const colCh = show(chars[state.fwd.tokens[col]]);
    box.innerHTML = [
        `“${rowCh}” (pos ${row}) asks “${colCh}” (pos ${col})`,
        `q·k / √16      = <b>${raw.toFixed(3)}</b>`,
        `softmax share  = <b>${(p * 100).toFixed(1)}%</b> of pos ${row}'s attention`,
    ].join('\n');
}

function renderTiles() {
    if (!state.fwd) return;
    const d = state.model.d;
    const i = state.selected;
    drawHeat($('tile-tok'), state.fwd.tokEmb.subarray(i * d, (i + 1) * d), 8, 8, { signed: true });
    drawHeat($('tile-pos'), state.fwd.posEmb.subarray(i * d, (i + 1) * d), 8, 8, { signed: true });
    const mlp = state.fwd.layers[state.layer].mlpAct;
    drawHeat($('tile-mlp'), mlp.subarray(i * 4 * d, i * 4 * d + 256), 16, 16, { signed: true });
}

// ---------- embedding map (PCA) ----------
function computePca() {
    const m = state.model;
    const V = m.vocab;
    const d = m.d;
    const emb = m.t.tok_emb.data;
    const mean = new Float64Array(d);
    for (let v = 0; v < V; v++) for (let c = 0; c < d; c++) mean[c] += emb[v * d + c] / V;
    const X = new Float64Array(V * d);
    for (let v = 0; v < V; v++) for (let c = 0; c < d; c++) X[v * d + c] = emb[v * d + c] - mean[c];

    const cov = new Float64Array(d * d);
    for (let v = 0; v < V; v++) {
        for (let a = 0; a < d; a++) {
            const xa = X[v * d + a];
            if (xa === 0) continue;
            for (let b = 0; b < d; b++) cov[a * d + b] += xa * X[v * d + b];
        }
    }
    const power = (deflate) => {
        let vec = new Float64Array(d).fill(1 / Math.sqrt(d));
        for (let it = 0; it < 80; it++) {
            const next = new Float64Array(d);
            for (let a = 0; a < d; a++) {
                let s = 0;
                for (let b = 0; b < d; b++) s += cov[a * d + b] * vec[b];
                next[a] = s;
            }
            if (deflate) {
                let dot = 0;
                for (let c = 0; c < d; c++) dot += next[c] * deflate[c];
                for (let c = 0; c < d; c++) next[c] -= dot * deflate[c];
            }
            let norm = 0;
            for (const x of next) norm += x * x;
            norm = Math.sqrt(norm) || 1;
            for (let c = 0; c < d; c++) next[c] /= norm;
            vec = next;
        }
        return vec;
    };
    const v1 = power(null);
    const v2 = power(v1);
    const coords = [];
    for (let v = 0; v < V; v++) {
        let x = 0;
        let y = 0;
        for (let c = 0; c < d; c++) {
            x += X[v * d + c] * v1[c];
            y += X[v * d + c] * v2[c];
        }
        coords.push([x, y]);
    }
    state.pca = coords;
}

function buildEmbMap() {
    const canvas = $('emb-map');
    const m = state.model;
    let hover = -1;

    const draw = () => {
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        let maxAbs = 1e-9;
        for (const [x, y] of state.pca) maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
        const sx = (x) => w / 2 + (x / maxAbs) * (w / 2 - 34);
        const sy = (y) => h / 2 - (y / maxAbs) * (h / 2 - 30);

        ctx.font = '17px "Plex Mono", ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        state.pca.forEach(([x, y], v) => {
            const ch = m.chars[v];
            const isHover = v === hover;
            ctx.fillStyle = isHover ? '#eeebe3'
                : /[a-z]/.test(ch) ? '#86c3d6'
                : /[A-Z]/.test(ch) ? '#6a9955'
                : /[0-9]/.test(ch) ? '#c2a24e'
                : '#8f8b83';
            ctx.fillText(show(ch), sx(x), sy(y));
        });

        if (hover >= 0) {
            // nearest neighbors by true 64-d cosine
            const sims = [];
            for (let v = 0; v < m.vocab; v++) {
                if (v !== hover) sims.push([v, cosine(m, hover, v)]);
            }
            sims.sort((a, b) => b[1] - a[1]);
            const list = sims.slice(0, 5).map(([v, c]) => `${show(m.chars[v])} ${c.toFixed(2)}`).join('  ');
            ctx.fillStyle = '#8f8b83';
            ctx.textAlign = 'left';
            ctx.fillText(`“${show(m.chars[hover])}” neighbors: ${list}`, 12, h - 14);
        }
    };

    canvas.addEventListener('pointermove', (e) => {
        const r = canvas.getBoundingClientRect();
        const px = ((e.clientX - r.left) / r.width) * canvas.width;
        const py = ((e.clientY - r.top) / r.height) * canvas.height;
        let best = -1;
        let bestD = 400;
        let maxAbs = 1e-9;
        for (const [x, y] of state.pca) maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
        state.pca.forEach(([x, y], v) => {
            const dx = canvas.width / 2 + (x / maxAbs) * (canvas.width / 2 - 34) - px;
            const dy = canvas.height / 2 - (y / maxAbs) * (canvas.height / 2 - 30) - py;
            const dd = dx * dx + dy * dy;
            if (dd < bestD) { bestD = dd; best = v; }
        });
        if (best !== hover) { hover = best; draw(); }
    });
    canvas.addEventListener('pointerleave', () => { hover = -1; draw(); });
    draw();
}

function cosine(m, a, b) {
    const d = m.d;
    const e = m.t.tok_emb.data;
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let c = 0; c < d; c++) {
        const xa = e[a * d + c];
        const xb = e[b * d + c];
        dot += xa * xb;
        na += xa * xa;
        nb += xb * xb;
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

// ---------- cosine explorer ----------
function buildCosExplorer() {
    const m = state.model;
    const selA = $('cos-a');
    const selB = $('cos-b');
    for (const sel of [selA, selB]) {
        for (let v = 0; v < m.vocab; v++) {
            const o = document.createElement('option');
            o.value = String(v);
            o.textContent = `${show(m.chars[v])}  (id ${v})`;
            sel.append(o);
        }
    }
    selA.value = String(m.stoi.e ?? 1);
    selB.value = String(m.stoi.o ?? 2);
    const draw = () => drawCos(parseInt(selA.value, 10), parseInt(selB.value, 10));
    selA.addEventListener('change', draw);
    selB.addEventListener('change', draw);
    draw();
}

function drawCos(a, b) {
    const m = state.model;
    const canvas = $('cos-view');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const cos = cosine(m, a, b);
    const theta = Math.acos(Math.min(1, Math.max(-1, cos)));
    const e = m.t.tok_emb.data;
    const norm = (v) => {
        let s = 0;
        for (let c = 0; c < m.d; c++) s += e[v * m.d + c] ** 2;
        return Math.sqrt(s);
    };
    const na = norm(a);
    const nb = norm(b);
    const scale = (Math.min(w, h) * 0.42) / Math.max(na, nb);
    const cx = w * 0.5;
    const cy = h * 0.78;

    const arrow = (angle, len, color, label) => {
        const x = cx + Math.cos(angle) * len * scale;
        const y = cy - Math.sin(angle) * len * scale;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.font = '18px "Plex Mono", ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(label, x, y - 14);
    };
    const base = (Math.PI - theta) / 2;
    arrow(base + theta, na, '#6a9955', show(m.chars[a]));
    arrow(base, nb, '#c25b4e', show(m.chars[b]));

    // the true angle between them
    ctx.strokeStyle = '#5f5c55';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(cx, cy, 44, -(base + theta), -base);
    ctx.stroke();
    ctx.fillStyle = '#8f8b83';
    ctx.font = '17px "Plex Mono", ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${(theta * 180 / Math.PI).toFixed(0)}°`, cx, cy - 56);

    let dot = 0;
    for (let c = 0; c < m.d; c++) dot += e[a * m.d + c] * e[b * m.d + c];
    $('cos-read').innerHTML = [
        `cos θ = <b>${cos.toFixed(3)}</b>   (θ = ${(theta * 180 / Math.PI).toFixed(1)}°)`,
        `a·b = |a||b|cos θ = ${na.toFixed(2)} × ${nb.toFixed(2)} × ${cos.toFixed(3)} = <b>${dot.toFixed(3)}</b>`,
    ].join('\n');
}

// ---------- loop + wiring ----------
function loop(now) {
    if (state.playing && now - state.lastStep > 1000 / state.speed) {
        state.lastStep = now;
        stepToken();
    }
    requestAnimationFrame(loop);
}

$('step').addEventListener('click', () => {
    state.playing = false;
    $('autoplay').textContent = 'generate';
    stepToken();
});
$('autoplay').addEventListener('click', () => {
    state.playing = !state.playing;
    $('autoplay').textContent = state.playing ? 'pause' : 'generate';
});
$('reset').addEventListener('click', resetText);
$('prompt').addEventListener('change', resetText);
$('temp').addEventListener('input', (e) => {
    state.temperature = parseFloat(e.target.value);
    $('temp-out').textContent = state.temperature.toFixed(1);
    if (!state.playing) refresh(false);
});
$('topk').addEventListener('change', (e) => {
    state.topK = parseInt(e.target.value, 10);
    if (!state.playing) refresh(false);
});
$('speed').addEventListener('input', (e) => {
    state.speed = parseInt(e.target.value, 10);
    $('speed-out').textContent = `${state.speed}/s`;
});
window.addEventListener('resize', drawArcs);
