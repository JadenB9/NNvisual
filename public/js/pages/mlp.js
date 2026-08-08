import { mulberry32, shuffle } from '../engine/rng.js';
import { MLP } from '../engine/mlp.js';
import { fetchWeights } from '../engine/weights.js';
import { fetchDataset, preprocessDrawing } from '../engine/mnist.js';
import { DrawBox } from '../ui/drawbox.js';
import { drawHeat } from '../ui/heat.js';
import { Bars } from '../ui/bars.js';
import { LineChart } from '../ui/charts.js';
import { Pipeline } from './mlp-pipeline.js';

const $ = (id) => document.getElementById(id);

const state = {
    train: null,          // mnist train dataset
    test: null,           // mnist test dataset
    ref: null,            // pretrained weights file
    net: null,
    hiddenSizes: [32, 16],
    usingRef: true,
    netSeed: 1,
    lr: 0.1,
    batch: 32,
    source: 'mnist',
    playing: false,
    stopAtEpochEnd: false,
    epoch: 0,
    cursor: 0,
    order: [],
    input: new Float32Array(784),
    userData: [],          // {label, px: number[784] 0..255}
    pickedLabel: null,
    probe: null,           // gradient-descent probe {alphas, losses, lr}
    lastBatch: null,       // {x, y, n} for the probe
};

const STORE_KEY = 'nnvisual-userdata-v1';

// ---------- dom ----------
const pipeline = new Pipeline($('pipeline'), $('inspector'), $('insp-heat'), $('insp-text'));
const bars = new Bars($('bars'), [...Array(10).keys()].map(String));
const lossChart = new LineChart($('chart-loss'), [{ color: '#86c3d6', label: 'loss' }]);
const accChart = new LineChart($('chart-acc'), [{ color: '#6a9955', label: 'test acc' }], { yMax: 1 });

const draw = new DrawBox($('draw'), { onchange: onDrawChange });

// ---------- boot ----------
(async function boot() {
    const fill = $('load-fill');
    const text = $('load-text');
    try {
        const seen = [0, 0, 0];
        const track = (i) => (got, total) => {
            seen[i] = total ? got / total : 0;
            const pct = Math.round(((seen[0] + seen[1] + seen[2]) / 3) * 100);
            fill.style.width = `${pct}%`;
            text.textContent = `fetching mnist + weights… ${pct}%`;
        };
        const [train, test, ref] = await Promise.all([
            fetchDataset('./data/mnist-train.bin', track(0)),
            fetchDataset('./data/mnist-test.bin', track(1)),
            fetchWeights('./data/weights/mlp-ref.bin', track(2)),
        ]);
        state.train = train;
        state.test = test;
        state.ref = ref;
    } catch (err) {
        text.textContent = `failed to load data: ${err.message}`;
        throw err;
    }
    $('loader').classList.add('done');

    loadUserData();
    buildRefNet();
    renderLayerEditor();
    renderLabelPick();
    renderGallery();
    fillSamples();
    onDrawChange(draw.value);
    updateStatus();
    requestAnimationFrame(frame);
})();

function buildRefNet() {
    const sizes = state.ref.meta.sizes;
    state.hiddenSizes = sizes.slice(1, -1);
    state.net = new MLP(sizes, { hidden: state.ref.meta.hidden }, mulberry32(state.netSeed));
    state.net.loadTensors(state.ref.tensors);
    state.usingRef = true;
    resetTrainingState();
}

function freshNet() {
    state.net = new MLP([784, ...state.hiddenSizes, 10], { hidden: 'relu' }, mulberry32(++state.netSeed));
    state.usingRef = false;
    resetTrainingState();
}

function resetTrainingState() {
    state.epoch = 0;
    state.cursor = 0;
    state.order = [];
    state.playing = false;
    state.probe = null;
    $('train').textContent = 'train';
    lossChart.reset();
    accChart.reset();
    $('shape-note').textContent = state.net.sizes.join(' → ');
    $('bowl-lr-out').textContent = state.lr.toFixed(2);
    renderWeightGallery();
    drawBowl();
    classify();
    updateStats();
    updateStatus();
}

// ---------- input / classify ----------
function onDrawChange(value) {
    state.input = preprocessDrawing(value);
    drawHeat($('preview'), state.input, 28, 28, { max: 1 });
    $('add-example').disabled = draw.empty || state.pickedLabel == null;
    classify(true);
}

function classify(pulse = false) {
    if (!state.net) return;
    const fwd = state.net.forwardBatch(state.input, 1);
    state.lastFwd = fwd;
    const nOut = 10;
    const logits = fwd.acts[fwd.acts.length - 1];
    const showLogits = $('show-logits').checked;
    let best = 0;
    for (let o = 1; o < nOut; o++) if (fwd.out[o] > fwd.out[best]) best = o;

    if (showLogits) {
        // map logits into [0,1] for the bar widths, print the true values
        let lo = Infinity;
        let hi = -Infinity;
        for (let o = 0; o < nOut; o++) { lo = Math.min(lo, logits[o]); hi = Math.max(hi, logits[o]); }
        const span = hi - lo || 1;
        const vals = [...Array(nOut).keys()].map((o) => (logits[o] - lo) / span);
        bars.update(vals, best);
        for (let o = 0; o < nOut; o++) bars.rows[o].value.textContent = logits[o].toFixed(2);
    } else {
        bars.update(Array.from(fwd.out.slice(0, 10)), best);
    }
    $('pred').textContent = draw.empty && !state.loadedSample ? '–' : String(best);

    pipeline.setState(state.net, { acts: fwd.acts, zs: fwd.zs }, state.input);
    if (pulse) pipeline.startPulse();
    pipeline.draw();
}

// ---------- samples strip ----------
function fillSamples() {
    const strip = $('samples');
    strip.textContent = '';
    const rand = mulberry32(Date.now() & 0xffff);
    const used = new Set();
    for (let d = 0; d < 10; d++) {
        // find a random test image with label d
        let idx = -1;
        for (let tries = 0; tries < 200; tries++) {
            const i = Math.floor(rand() * state.test.count);
            if (state.test.labels[i] === d && !used.has(i)) { idx = i; break; }
        }
        if (idx < 0) continue;
        used.add(idx);
        const c = document.createElement('canvas');
        c.width = 28;
        c.height = 28;
        c.style.width = '34px';
        c.style.height = '34px';
        c.className = 'pix';
        c.title = `real test digit: ${d}`;
        const img = state.test.image(idx);
        drawHeat(c, img, 28, 28, { max: 1 });
        c.addEventListener('click', () => {
            state.loadedSample = true;
            draw.setValue(img);
            state.loadedSample = false;
        });
        strip.append(c);
    }
}

// ---------- training data assembly ----------
function trainingSet() {
    if (state.source === 'mnist') return { kind: 'mnist', count: state.train.count };
    const user = state.userData;
    if (state.source === 'user') return { kind: 'user', count: user.length };
    return { kind: 'both', count: state.train.count + user.length };
}

function fillTrainBatch(x, y, batchSize) {
    const set = trainingSet();
    if (set.count === 0) return 0;
    if (state.order.length !== set.count) {
        state.order = shuffle([...Array(set.count).keys()], mulberry32(state.epoch + 99));
        state.cursor = 0;
    }
    const n = Math.min(batchSize, set.count - state.cursor);
    for (let b = 0; b < n; b++) {
        const gi = state.order[state.cursor + b];
        if (set.kind === 'user' || (set.kind === 'both' && gi >= state.train.count)) {
            const ex = state.userData[set.kind === 'user' ? gi : gi - state.train.count];
            for (let p = 0; p < 784; p++) x[b * 784 + p] = ex.px[p] / 255;
            y[b] = ex.label;
        } else {
            const off = gi * 784;
            for (let p = 0; p < 784; p++) x[b * 784 + p] = state.train.pixels[off + p] / 255;
            y[b] = state.train.labels[gi];
        }
    }
    state.cursor += n;
    return n;
}

// ---------- the training loop ----------
const batchX = new Float32Array(64 * 784);
const batchY = new Uint8Array(64);
let lastLoss = 0;
let epochLosses = [];

function frame() {
    const budgetMs = 11;
    if (state.playing) {
        const start = performance.now();
        while (performance.now() - start < budgetMs) {
            const n = fillTrainBatch(batchX, batchY, state.batch);
            if (n === 0) { setPlaying(false); break; }
            lastLoss = state.net.trainBatch(batchX, batchY, n, state.lr);
            epochLosses.push(lastLoss);
            state.lastBatch = { x: batchX.slice(0, n * 784), y: batchY.slice(0, n), n };
            if (state.cursor >= trainingSet().count) {
                endOfEpoch();
                if (state.stopAtEpochEnd) { state.stopAtEpochEnd = false; setPlaying(false); }
                break;
            }
        }
        classify();
        updateStats();
    }
    if (pipeline.tick()) pipeline.draw();
    requestAnimationFrame(frame);
}

function endOfEpoch() {
    state.epoch++;
    state.cursor = 0;
    state.order = [];
    const meanLoss = epochLosses.reduce((a, b) => a + b, 0) / Math.max(1, epochLosses.length);
    epochLosses = [];
    const acc = testAccuracy();
    lossChart.push(meanLoss);
    accChart.push(acc);
    if (state.epoch % 2 === 0) renderWeightGallery();
    probeLoss();
    updateStatus();
}

function testAccuracy() {
    // always measured on the held-out MNIST test set
    const n = 500; // half the test set per epoch keeps it snappy
    const x = new Float32Array(n * 784);
    const y = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
        const off = i * 784;
        for (let p = 0; p < 784; p++) x[i * 784 + p] = state.test.pixels[off + p] / 255;
        y[i] = state.test.labels[i];
    }
    return state.net.accuracy(x, y, n);
}

function userAccuracy() {
    if (!state.userData.length) return null;
    const n = state.userData.length;
    const x = new Float32Array(n * 784);
    const y = new Uint8Array(n);
    state.userData.forEach((ex, i) => {
        for (let p = 0; p < 784; p++) x[i * 784 + p] = ex.px[p] / 255;
        y[i] = ex.label;
    });
    return state.net.accuracy(x, y, n);
}

function setPlaying(on) {
    if (on && trainingSet().count === 0) {
        $('data-note').textContent = 'no examples yet — draw some digits below first.';
        return;
    }
    state.playing = on;
    $('train').textContent = on ? 'pause' : 'train';
    if (on) state.usingRef = false;
}

// ---------- gradient descent probe (the "bowl") ----------
function probeLoss() {
    if (!state.lastBatch || !state.net.lastGrads) return;
    const { gw, gb } = state.net.lastGrads;
    const net = state.net;
    const probeN = Math.min(256, state.lastBatch.n * 8, trainingSet().count);
    // reuse the last batch as the probe set (honest: it's the loss surface
    // the step was actually taken on)
    const { x, y, n } = state.lastBatch;

    // save weights
    const savedW = net.weights.map((w) => w.slice());
    const savedB = net.biases.map((b) => b.slice());

    const span = state.lr * 2.5;
    const alphas = [];
    const losses = [];
    for (let k = 0; k <= 24; k++) {
        const alpha = -span * 0.15 + (k / 24) * span * 1.15;
        for (let l = 0; l < net.depth; l++) {
            const w = net.weights[l];
            const sw = savedW[l];
            const g = gw[l];
            for (let i = 0; i < w.length; i++) w[i] = sw[i] - alpha * g[i];
            const b = net.biases[l];
            const sb = savedB[l];
            const gbl = gb[l];
            for (let i = 0; i < b.length; i++) b[i] = sb[i] - alpha * gbl[i];
        }
        alphas.push(alpha);
        losses.push(net.evalLoss(x, y, n));
    }
    // restore
    for (let l = 0; l < net.depth; l++) {
        net.weights[l].set(savedW[l]);
        net.biases[l].set(savedB[l]);
    }
    state.probe = { alphas, losses, probeN };
    drawBowl();
}

function drawBowl() {
    const canvas = $('bowl');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.font = '19px "Plex Mono", ui-monospace, monospace';
    if (!state.probe) {
        ctx.fillStyle = '#5f5c55';
        ctx.textAlign = 'center';
        ctx.fillText('train a step to measure the landscape', w / 2, h / 2 + 6);
        return;
    }
    const { alphas, losses } = state.probe;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of losses) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = 26;
    const xAt = (a) => pad + ((a - alphas[0]) / (alphas[alphas.length - 1] - alphas[0])) * (w - 2 * pad);
    const yAt = (v) => h - pad - ((v - lo) / (hi - lo || 1)) * (h - 2 * pad);

    ctx.strokeStyle = '#86c3d6';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    alphas.forEach((a, i) => {
        if (i === 0) ctx.moveTo(xAt(a), yAt(losses[i]));
        else ctx.lineTo(xAt(a), yAt(losses[i]));
    });
    ctx.stroke();

    const lerp = (alpha) => {
        for (let i = 1; i < alphas.length; i++) {
            if (alpha <= alphas[i]) {
                const f = (alpha - alphas[i - 1]) / (alphas[i] - alphas[i - 1]);
                return losses[i - 1] + f * (losses[i] - losses[i - 1]);
            }
        }
        return losses[losses.length - 1];
    };

    // marker: where we are (alpha 0) and where the chosen lr lands
    const mark = (alpha, color, label) => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(xAt(alpha), yAt(lerp(alpha)), 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.textAlign = 'center';
        ctx.fillText(label, xAt(alpha), yAt(lerp(alpha)) - 12);
    };
    mark(0, '#eeebe3', 'here');
    const ghostLr = parseFloat($('bowl-lr').dataset.value || state.lr);
    mark(ghostLr, '#6a9955', `step lr=${ghostLr.toFixed(2)}`);

    ctx.fillStyle = '#5f5c55';
    ctx.textAlign = 'left';
    ctx.fillText('→ along the gradient', pad, h - 6);
}

// ---------- weight gallery ----------
function renderWeightGallery() {
    const box = $('weight-gallery');
    box.textContent = '';
    const n1 = state.net.sizes[1];
    const w = state.net.weights[0];
    for (let j = 0; j < n1; j++) {
        const c = document.createElement('canvas');
        c.width = 28;
        c.height = 28;
        c.style.width = '42px';
        c.style.height = '42px';
        c.className = 'pix';
        c.title = `hidden1 #${j}`;
        drawHeat(c, w.subarray(j * 784, (j + 1) * 784), 28, 28, { signed: true });
        box.append(c);
    }
}

// ---------- layer editor ----------
function renderLayerEditor() {
    const box = $('layers');
    box.textContent = '';
    state.hiddenSizes.forEach((count, idx) => {
        const chip = document.createElement('div');
        chip.className = 'layer-chip';
        const label = document.createElement('span');
        label.className = 'lc-label';
        label.textContent = `hidden ${idx + 1}`;
        const minus = mkBtn('−', () => {
            if (state.hiddenSizes[idx] > 2) {
                state.hiddenSizes[idx]--;
                renderLayerEditor();
                freshNet();
            }
        });
        const value = document.createElement('span');
        value.className = 'lc-count';
        value.textContent = String(count);
        const plus = mkBtn('+', () => {
            if (state.hiddenSizes[idx] < 48) {
                state.hiddenSizes[idx]++;
                renderLayerEditor();
                freshNet();
            }
        });
        chip.append(label, minus, value, plus);
        box.append(chip);
    });
    const refSizes = state.ref.meta.sizes.slice(1, -1);
    $('load-ref').disabled = JSON.stringify(state.hiddenSizes) !== JSON.stringify(refSizes);
}

function mkBtn(text, fn) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'lc-btn';
    b.textContent = text;
    b.addEventListener('click', fn);
    return b;
}

// ---------- user dataset ----------
function loadUserData() {
    try {
        const raw = localStorage.getItem(STORE_KEY);
        if (raw) {
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed.examples)) state.userData = parsed.examples;
        }
    } catch { /* fresh start */ }
}

function saveUserData() {
    try {
        localStorage.setItem(STORE_KEY, JSON.stringify({ v: 1, examples: state.userData }));
    } catch {
        $('data-note').textContent = 'storage full — export and trim your dataset.';
    }
}

function renderLabelPick() {
    const box = $('label-pick');
    box.textContent = '';
    for (let d = 0; d < 10; d++) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'seg-btn';
        b.textContent = String(d);
        b.addEventListener('click', () => {
            state.pickedLabel = d;
            box.querySelectorAll('.seg-btn').forEach((x) => x.classList.toggle('is-active', x === b));
            $('add-example').textContent = `add drawing as “${d}”`;
            $('add-example').disabled = draw.empty;
        });
        box.append(b);
    }
}

function renderGallery() {
    const box = $('gallery');
    box.textContent = '';
    const byLabel = new Map();
    state.userData.forEach((ex, i) => {
        if (!byLabel.has(ex.label)) byLabel.set(ex.label, []);
        byLabel.get(ex.label).push({ ex, i });
    });
    for (let d = 0; d < 10; d++) {
        const list = byLabel.get(d) || [];
        if (!list.length) continue;
        const row = document.createElement('div');
        row.className = 'g-row';
        const label = document.createElement('span');
        label.className = 'g-label';
        label.textContent = `${d} ×${list.length}`;
        const thumbs = document.createElement('div');
        thumbs.className = 'g-thumbs';
        for (const { ex, i } of list) {
            const c = document.createElement('canvas');
            c.width = 28;
            c.height = 28;
            c.style.width = '30px';
            c.style.height = '30px';
            c.className = 'pix';
            c.title = 'click to delete';
            const f = new Float32Array(784);
            for (let p = 0; p < 784; p++) f[p] = ex.px[p] / 255;
            drawHeat(c, f, 28, 28, { max: 1 });
            c.addEventListener('click', () => {
                state.userData.splice(i, 1);
                saveUserData();
                renderGallery();
                updateStatus();
            });
            thumbs.append(c);
        }
        row.append(label, thumbs);
        box.append(row);
    }
    if (!state.userData.length) {
        const p = document.createElement('p');
        p.className = 'note';
        p.textContent = 'no examples yet — your digits will appear here.';
        box.append(p);
    }
}

// ---------- status ----------
function updateStatus() {
    const you = state.userData.length ? ` · you: ${state.userData.length} digits` : '';
    const net = state.usingRef
        ? `pretrained ${(state.ref.meta.testAcc * 100).toFixed(1)}%`
        : `your net · epoch ${state.epoch}`;
    $('net-status').textContent = `mnist 6,000 train · 1,000 test${you} · ${net}`;
}

function updateStats() {
    const el = $('train-stats');
    const acc = accChart.series[0].data;
    const uacc = userAccuracy();
    el.innerHTML = '';
    const lines = [
        `epoch <b>${state.epoch}</b>`,
        `last batch loss <b>${lastLoss ? lastLoss.toFixed(3) : '–'}</b>`,
        `test accuracy <b>${acc.length ? (acc[acc.length - 1] * 100).toFixed(1) + '%' : '–'}</b>`,
    ];
    if (uacc != null && state.source !== 'mnist') {
        lines.push(`your digits <b>${(uacc * 100).toFixed(0)}%</b>`);
    }
    for (const l of lines) {
        const div = document.createElement('div');
        div.innerHTML = l;
        el.append(div);
    }
}

// ---------- wiring ----------
$('clear').addEventListener('click', () => draw.clear());
$('undo').addEventListener('click', () => draw.undo());
$('shuffle-samples').addEventListener('click', fillSamples);
$('show-logits').addEventListener('change', () => classify());

$('train').addEventListener('click', () => setPlaying(!state.playing));
$('step-epoch').addEventListener('click', () => {
    state.stopAtEpochEnd = true;
    setPlaying(true);
});
$('reset-net').addEventListener('click', freshNet);
$('load-ref').addEventListener('click', buildRefNet);

$('lr').addEventListener('change', (e) => { state.lr = parseFloat(e.target.value); });
$('batch').addEventListener('change', (e) => { state.batch = parseInt(e.target.value, 10); });
$('source').addEventListener('change', (e) => {
    state.source = e.target.value;
    state.order = [];
    state.cursor = 0;
    updateStats();
});

$('add-layer').addEventListener('click', () => {
    if (state.hiddenSizes.length < 3) {
        state.hiddenSizes.push(8);
        renderLayerEditor();
        freshNet();
    }
});
$('remove-layer').addEventListener('click', () => {
    if (state.hiddenSizes.length > 1) {
        state.hiddenSizes.pop();
        renderLayerEditor();
        freshNet();
    }
});

$('replay-bp').addEventListener('click', () => {
    if (!state.net.lastGrads) {
        $('data-note').textContent = '';
        alertNote('train at least one batch first — there are no gradients yet.');
        return;
    }
    pipeline.startBackprop();
});

function alertNote(msg) {
    $('arch-note').textContent = msg;
    setTimeout(() => { $('arch-note').textContent = 'editing the architecture resets to fresh random weights.'; }, 3200);
}

$('bowl-lr').addEventListener('input', (e) => {
    const f = e.target.value / 100;
    const v = f * state.lr * 2.5;
    e.target.dataset.value = String(v);
    $('bowl-lr-out').textContent = v.toFixed(2);
    drawBowl();
});

$('add-example').addEventListener('click', () => {
    if (state.pickedLabel == null || draw.empty) return;
    const px = new Array(784);
    for (let p = 0; p < 784; p++) px[p] = Math.round(state.input[p] * 255);
    if (state.userData.length >= 500) {
        $('data-note').textContent = 'capped at 500 examples — delete some first.';
        return;
    }
    state.userData.push({ label: state.pickedLabel, px });
    saveUserData();
    renderGallery();
    updateStatus();
    $('data-note').textContent = `added a “${state.pickedLabel}” — ${state.userData.length} total. Draw the next one!`;
    draw.clear();
});

$('clear-data').addEventListener('click', () => {
    state.userData = [];
    saveUserData();
    renderGallery();
    updateStatus();
});

$('export-data').addEventListener('click', () => {
    const io = $('data-io');
    io.hidden = false;
    io.value = JSON.stringify({ v: 1, examples: state.userData });
    io.select();
    $('data-note').textContent = 'copy this JSON somewhere safe.';
});

$('import-data').addEventListener('click', () => {
    const io = $('data-io');
    if (io.hidden) {
        io.hidden = false;
        io.value = '';
        io.placeholder = 'paste exported JSON here, then press import again';
        return;
    }
    try {
        const parsed = JSON.parse(io.value);
        if (!Array.isArray(parsed.examples)) throw new Error('bad shape');
        const clean = parsed.examples.filter((e) => Number.isInteger(e.label)
            && e.label >= 0 && e.label <= 9 && Array.isArray(e.px) && e.px.length === 784);
        state.userData = clean.slice(0, 500);
        saveUserData();
        renderGallery();
        updateStatus();
        io.hidden = true;
        $('data-note').textContent = `imported ${state.userData.length} examples.`;
    } catch {
        $('data-note').textContent = 'could not parse that JSON.';
    }
});

window.addEventListener('resize', () => pipeline.draw());
