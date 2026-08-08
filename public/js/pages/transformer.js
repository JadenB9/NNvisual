import { fetchWeights } from '../engine/weights.js';
import { Transformer } from '../engine/transformer.js';
import { mulberry32 } from '../engine/rng.js';
import { drawHeat } from '../ui/heat.js';
import { Bars } from '../ui/bars.js';
import { nucleus, argmax, rankOf, keptCount } from './sampling.js';

const $ = (id) => document.getElementById(id);
const show = (ch) => (ch === '\n' ? '⏎' : ch === ' ' ? '␣' : ch);

const MEAN = -1;               // head picker value for "average of this layer's heads"
const FALLBACK_PROMPT = 'ROMEO:';

const state = {
    model: null,
    tokens: [],        // full generated sequence (token ids)
    promptLen: 0,
    fwd: null,         // capture for the current context window
    layer: 1,
    head: 0,           // head index, or MEAN
    selected: 0,       // selected position within the window
    pair: null,        // [row, col] in the matrix
    playing: false,
    speed: 4,
    lastStep: 0,
    temperature: 0.8,
    topK: 10,
    topP: 0,           // 0 = off
    greedy: false,
    lastProbs: null,   // the distribution the bars are showing
    lastLogits: null,  // raw logits behind it, before temperature
    sampled: null,     // token drawn from lastProbs, or null if nothing was drawn from it
    attnCache: null,
    rand: mulberry32(Date.now() & 0xffffff),
    pca: null,
};

let nextBars = null;
let highlightEmb = () => {};   // assigned by buildEmbMap

// The vocabulary contains "&" and "'", so page text is built from nodes
// rather than innerHTML.
function bold(text) {
    const b = document.createElement('b');
    b.textContent = text;
    return b;
}

function setLine(el, parts) {
    el.textContent = '';
    for (const part of parts) {
        el.append(typeof part === 'string' ? document.createTextNode(part) : part);
    }
}

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
    const m = state.model;
    if (!m) return;
    const typed = [...$('prompt').value];
    let tokens = m.encode($('prompt').value);
    const note = $('prompt-note');
    note.textContent = '';
    if (tokens.length === 0) {
        tokens = m.encode(FALLBACK_PROMPT);
        note.textContent = typed.length === 0
            ? `empty prompt — starting from ${FALLBACK_PROMPT}`
            : `none of those characters are in the model's ${m.vocab}-character alphabet — starting from ${FALLBACK_PROMPT}`;
    } else if (tokens.length < typed.length) {
        const dropped = typed.length - tokens.length;
        note.textContent = dropped === 1
            ? `1 unknown character was dropped — the model only knows ${m.vocab} of them.`
            : `${dropped} unknown characters were dropped — the model only knows ${m.vocab} of them.`;
    }
    state.tokens = tokens;
    state.promptLen = tokens.length;
    state.playing = false;
    $('autoplay').textContent = 'generate';
    refresh(false);
}

// One real forward pass, shaped by the current controls. Greedy ignores
// temperature/top-k/top-p — none of them can move which score is largest — so
// the bars then show the model's own distribution.
function distribution() {
    const m = state.model;
    const ctx = state.tokens.slice(-m.block);
    const { probs, logits, fwd } = m.nextDistribution(
        ctx,
        state.greedy ? 1 : state.temperature,
        state.greedy ? 0 : state.topK
    );
    return { probs: state.greedy ? probs : nucleus(probs, state.topP), logits, fwd };
}

function stepToken() {
    if (!state.model || !state.tokens.length) return;
    const { probs, logits, fwd } = distribution();
    const tok = state.greedy ? argmax(probs) : state.model.sample(probs, state.rand);
    state.tokens.push(tok);
    state.fwd = fwd;
    state.lastProbs = probs;
    state.lastLogits = logits;
    state.sampled = tok;
    refresh(true);
}

function refresh(fromStep) {
    if (!state.model || !state.tokens.length) return;
    if (!fromStep) {
        const { probs, logits, fwd } = distribution();
        state.fwd = fwd;
        state.lastProbs = probs;
        state.lastLogits = logits;
        state.sampled = null;   // these bars produced nothing yet
    }
    state.pair = null;
    setSelected(state.fwd.T - 1);
    renderText();
    renderContextUse();
    renderChips();
    renderNextBars();
    renderAttn();
    renderTiles();
    renderPair();
}

// the selected position drives the arcs, the tiles, and the ring on the map
function setSelected(i) {
    state.selected = i;
    highlightEmb(state.fwd.tokens[i], i);
}

function renderContextUse() {
    const m = state.model;
    const used = Math.min(state.tokens.length, m.block);
    $('ctx-use').textContent = `${used}/${m.block} chars in context`;
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
            setSelected(i);
            renderChips();
            renderAttn();
            renderTiles();
        });
        box.append(b);
    });
    requestAnimationFrame(drawArcs);
}

function renderNextBars() {
    const m = state.model;
    const probs = state.lastProbs;
    const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 10);
    nextBars.setLabels(order.map((i) => show(m.chars[i])));
    nextBars.update(order.map((i) => probs[i]), 0);

    // every number below is read out of the arrays the bars were drawn from
    const note = $('sample-note');
    const kept = keptCount(probs);
    if (state.sampled === null) {
        const top = order[0];
        setLine(note, [
            state.greedy ? 'greedy — the next character will be ' : 'most likely next: ',
            bold(show(m.chars[top])),
            ` (${(probs[top] * 100).toFixed(1)}%, ${kept} of ${m.vocab} characters still in play)`,
        ]);
        return;
    }
    const tok = state.sampled;
    const knobs = state.greedy
        ? ' — the largest score, no dice roll'
        : ` — temperature ${state.temperature.toFixed(1)}`
            + (state.topK ? `, top-k ${state.topK}` : '')
            + (state.topP ? `, top-p ${state.topP}` : '');
    setLine(note, [
        state.greedy ? 'took ' : 'sampled ',
        bold(show(m.chars[tok])),
        `: rank ${rankOf(probs, tok)} of the ${kept} kept, `,
        bold(`${(probs[tok] * 100).toFixed(1)}%`),
        ` of the final distribution, raw logit ${state.lastLogits[tok].toFixed(2)}`,
        knobs,
    ]);
}

// ---------- attention rendering ----------
function buildHeadPicker() {
    const box = $('head-pick');
    box.textContent = '';
    const add = (label, layer, head) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'seg-btn' + (layer === state.layer && head === state.head ? ' is-active' : '');
        b.textContent = label;
        b.addEventListener('click', () => {
            state.layer = layer;
            state.head = head;
            box.querySelectorAll('.seg-btn').forEach((x) => x.classList.toggle('is-active', x === b));
            renderAttn();
            renderPair();
        });
        box.append(b);
    };
    for (let l = 0; l < state.model.nLayer; l++) {
        for (let h = 0; h < state.model.nHead; h++) add(`L${l + 1}·H${h + 1}`, l, h);
        add(`L${l + 1}·mean`, l, MEAN);
    }
}

// The T×T map the matrix and the arcs both draw: one head's captured
// probabilities, or the true cell-by-cell average over the layer's heads.
// Rows of the average still sum to 1, since every row averaged does.
function attnProbs() {
    const cache = state.attnCache;
    if (cache && cache.fwd === state.fwd && cache.layer === state.layer && cache.head === state.head) {
        return cache.probs;
    }
    const heads = state.fwd.layers[state.layer].heads;
    let probs;
    if (state.head === MEAN) {
        const n = state.fwd.T * state.fwd.T;
        probs = new Float32Array(n);
        for (const head of heads) {
            for (let i = 0; i < n; i++) probs[i] += head.probs[i];
        }
        for (let i = 0; i < n; i++) probs[i] /= heads.length;
    } else {
        probs = heads[state.head].probs;
    }
    state.attnCache = { fwd: state.fwd, layer: state.layer, head: state.head, probs };
    return probs;
}

function renderAttn() {
    if (!state.fwd) return;
    const T = state.fwd.T;
    const probs = attnProbs();
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
    const probs = attnProbs();
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
    if (!state.fwd) return;
    const T = state.fwd.T;
    const r = e.currentTarget.getBoundingClientRect();
    const col = Math.floor(((e.clientX - r.left) / r.width) * T);
    const row = Math.floor(((e.clientY - r.top) / r.height) * T);
    if (col > row) return; // causal mask — the future is not an option
    state.pair = [row, col];
    setSelected(row);
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
    const cell = row * T + col;
    const heads = state.fwd.layers[state.layer].heads;
    const chars = state.model.chars;
    const rowCh = show(chars[state.fwd.tokens[row]]);
    const colCh = show(chars[state.fwd.tokens[col]]);
    const asks = `“${rowCh}” (pos ${row}) asks “${colCh}” (pos ${col})\n`;

    if (state.head === MEAN) {
        // averaged softmaxes have no single q·k, so show the heads instead
        setLine(box, [
            asks,
            `mean of ${heads.length} heads = `,
            bold(`${(attnProbs()[cell] * 100).toFixed(1)}%`),
            ` of pos ${row}'s attention\n`,
            heads.map((h, i) => `H${i + 1} ${(h.probs[cell] * 100).toFixed(1)}%`).join('   '),
            '\npick a single head for its q·k arithmetic',
        ]);
        return;
    }
    const head = heads[state.head];
    setLine(box, [
        asks,
        `q·k / √${state.model.d / state.model.nHead}      = `,
        bold(head.scores[cell].toFixed(3)),
        '\nsoftmax share  = ',
        bold(`${(head.probs[cell] * 100).toFixed(1)}%`),
        ` of pos ${row}'s attention`,
    ]);
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
    let marked = -1;      // token ringed by the chip / matrix selection
    let markedPos = -1;

    const project = () => {
        const w = canvas.width;
        const h = canvas.height;
        let maxAbs = 1e-9;
        for (const [x, y] of state.pca) maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
        return {
            sx: (x) => w / 2 + (x / maxAbs) * (w / 2 - 34),
            sy: (y) => h / 2 - (y / maxAbs) * (h / 2 - 30),
        };
    };

    const draw = () => {
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        const { sx, sy } = project();

        if (marked >= 0) {
            ctx.strokeStyle = '#86c3d6';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(sx(state.pca[marked][0]), sy(state.pca[marked][1]), 13, 0, Math.PI * 2);
            ctx.stroke();
        }

        ctx.font = '17px "Plex Mono", ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        state.pca.forEach(([x, y], v) => {
            const ch = m.chars[v];
            ctx.fillStyle = v === hover || v === marked ? '#eeebe3'
                : /[a-z]/.test(ch) ? '#86c3d6'
                : /[A-Z]/.test(ch) ? '#6a9955'
                : /[0-9]/.test(ch) ? '#c2a24e'
                : '#8f8b83';
            ctx.fillText(show(ch), sx(x), sy(y));
        });

        if (marked >= 0) {
            ctx.fillStyle = '#86c3d6';
            ctx.textAlign = 'left';
            ctx.font = '13px "Plex Mono", ui-monospace, monospace';
            ctx.fillText(`ringed: “${show(m.chars[marked])}”, the token selected at position ${markedPos}`, 12, 16);
            ctx.font = '17px "Plex Mono", ui-monospace, monospace';
        }
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

    highlightEmb = (token, pos) => {
        if (token === marked && pos === markedPos) return;
        marked = token;
        markedPos = pos;
        draw();
    };

    canvas.addEventListener('pointermove', (e) => {
        const r = canvas.getBoundingClientRect();
        const px = ((e.clientX - r.left) / r.width) * canvas.width;
        const py = ((e.clientY - r.top) / r.height) * canvas.height;
        const { sx, sy } = project();
        let best = -1;
        let bestD = 400;
        state.pca.forEach(([x, y], v) => {
            const dx = sx(x) - px;
            const dy = sy(y) - py;
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
    if (!state.model) return;
    state.playing = !state.playing;
    $('autoplay').textContent = state.playing ? 'pause' : 'generate';
});
$('reset').addEventListener('click', resetText);
$('preset').addEventListener('change', (e) => {
    if (e.target.value) $('prompt').value = e.target.value;
    resetText();
});
$('prompt').addEventListener('change', () => {
    syncPreset();
    resetText();
});
$('temp').addEventListener('input', (e) => {
    state.temperature = parseFloat(e.target.value);
    $('temp-out').textContent = state.temperature.toFixed(1);
    if (!state.playing) refresh(false);
});
$('topk').addEventListener('change', (e) => {
    state.topK = parseInt(e.target.value, 10);
    if (!state.playing) refresh(false);
});
$('topp').addEventListener('change', (e) => {
    state.topP = parseFloat(e.target.value);
    if (!state.playing) refresh(false);
});
$('greedy').addEventListener('click', () => setGreedy(!state.greedy));

// a typed prompt that matches no preset drops the picker back to "custom…"
function syncPreset() {
    const sel = $('preset');
    const typed = $('prompt').value;
    const known = [...sel.options].some((o) => o.value !== '' && o.value === typed);
    sel.value = known ? typed : '';
}

// Greedy always takes the largest score, which temperature, top-k and top-p
// cannot change — so they switch off while it is on.
function setGreedy(on) {
    state.greedy = on;
    const btn = $('greedy');
    btn.textContent = `greedy: ${on ? 'on' : 'off'}`;
    btn.classList.toggle('is-active', on);
    btn.setAttribute('aria-pressed', String(on));
    for (const name of ['temp', 'topk', 'topp']) {
        $(name).disabled = on;
        $(`${name}-field`).classList.toggle('is-off', on);
    }
    if (!state.playing) refresh(false);
}
$('speed').addEventListener('input', (e) => {
    state.speed = parseInt(e.target.value, 10);
    $('speed-out').textContent = `${state.speed}/s`;
});
window.addEventListener('resize', drawArcs);
