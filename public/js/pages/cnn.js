import { fetchWeights } from '../engine/weights.js';
import { fetchDataset } from '../engine/mnist.js';
import { ConvNet, conv2dForward } from '../engine/conv.js';
import { drawHeat } from '../ui/heat.js';
import { Bars } from '../ui/bars.js';
import { initInputPanel } from '../ui/inputpanel.js';

const $ = (id) => document.getElementById(id);

const state = {
    net: null,
    input: new Float32Array(784),
    stages: null,
    scan: { kernel: 0, pos: 0, playing: false, speed: 6 },
    playKernel: [0, -1, 0, -1, 4, -1, 0, -1, 0],
};

const bars = new Bars($('bars'), [...Array(10).keys()].map(String));
const mapCanvases = { maps1: [], pool1: [], maps2: [], pool2: [] };

// ---------- boot ----------
(async function boot() {
    const fill = $('load-fill');
    const text = $('load-text');
    let test;
    try {
        const seen = [0, 0];
        const track = (i) => (got, total) => {
            seen[i] = total ? got / total : 0;
            const pct = Math.round(((seen[0] + seen[1]) / 2) * 100);
            fill.style.width = `${pct}%`;
            text.textContent = `fetching weights + digits… ${pct}%`;
        };
        const [w, testSet] = await Promise.all([
            fetchWeights('./data/weights/cnn-ref.bin', track(0)),
            fetchDataset('./data/mnist-test.bin', track(1)),
        ]);
        state.net = new ConvNet().loadTensors(w.tensors);
        state.acc = w.meta.testAcc;
        test = testSet;
    } catch (err) {
        text.textContent = `failed to load: ${err.message}`;
        throw err;
    }
    $('loader').classList.add('done');
    $('net-status').textContent = `convnet · ${(state.acc * 100).toFixed(1)}% on held-out mnist`;

    buildStageCanvases();
    renderKernels();
    buildScanPicker();
    buildKernelEditor();

    initInputPanel({
        drawId: 'draw', clearId: 'clear', undoId: 'undo',
        samplesId: 'samples', shuffleId: 'shuffle-samples', previewId: 'preview',
        testSet: test,
        onInput: (input) => {
            state.input = input;
            runForward();
        },
    });

    requestAnimationFrame(scanFrame);
})();

// ---------- stage rendering ----------
function makeRow(containerId, count, size, css) {
    const box = $(containerId);
    box.textContent = '';
    const list = [];
    for (let i = 0; i < count; i++) {
        const c = document.createElement('canvas');
        c.width = size;
        c.height = size;
        c.style.width = `${css}px`;
        c.style.height = `${css}px`;
        c.className = 'pix';
        box.append(c);
        list.push(c);
    }
    return list;
}

function buildStageCanvases() {
    mapCanvases.maps1 = makeRow('maps1', 8, 26, 74);
    mapCanvases.pool1 = makeRow('pool1', 8, 13, 52);
    mapCanvases.maps2 = makeRow('maps2', 16, 11, 46);
    mapCanvases.pool2 = makeRow('pool2', 16, 5, 34);
    wireReceptiveHover();
}

function renderKernels() {
    const box = $('kernels1');
    box.textContent = '';
    for (let k = 0; k < 8; k++) {
        const c = document.createElement('canvas');
        c.width = 3;
        c.height = 3;
        c.style.width = '46px';
        c.style.height = '46px';
        c.className = 'pix';
        c.title = `kernel ${k + 1}`;
        drawHeat(c, state.net.c1w.subarray(k * 9, k * 9 + 9), 3, 3, { signed: true });
        box.append(c);
    }
}

function runForward() {
    if (!state.net) return;
    const s = state.net.forward(state.input);
    state.stages = s;

    let peak1 = 1e-9;
    for (const v of s.relu1) peak1 = Math.max(peak1, v);
    for (let k = 0; k < 8; k++) {
        drawHeat(mapCanvases.maps1[k], s.relu1.subarray(k * 676, (k + 1) * 676), 26, 26, { max: peak1 });
        drawHeat(mapCanvases.pool1[k], s.pool1.out.subarray(k * 169, (k + 1) * 169), 13, 13, { max: peak1 });
    }
    let peak2 = 1e-9;
    for (const v of s.relu2) peak2 = Math.max(peak2, v);
    for (let k = 0; k < 16; k++) {
        drawHeat(mapCanvases.maps2[k], s.relu2.subarray(k * 121, (k + 1) * 121), 11, 11, { max: peak2 });
        drawHeat(mapCanvases.pool2[k], s.pool2.out.subarray(k * 25, (k + 1) * 25), 5, 5, { max: peak2 });
    }
    drawHeat($('flat'), s.flat, 400, 1, { max: peak2 });

    let best = 0;
    for (let o = 1; o < 10; o++) if (s.probs[o] > s.probs[best]) best = o;
    bars.update(Array.from(s.probs), best);
    $('pred').textContent = state.input.some((v) => v > 0.02) ? String(best) : '–';

    state.scan.pos = 0; // restart the scan against fresh input
    drawScan();
    runPlayground();
}

// ---------- receptive field hover ----------
function wireReceptiveHover() {
    const spec = [
        { list: mapCanvases.maps1, grid: 26, rect: (x, y) => [x, y, 3, 3] },
        { list: mapCanvases.pool1, grid: 13, rect: (x, y) => [x * 2, y * 2, 4, 4] },
        { list: mapCanvases.maps2, grid: 11, rect: (x, y) => [x * 2, y * 2, 8, 8] },
        { list: mapCanvases.pool2, grid: 5, rect: (x, y) => [x * 4, y * 4, 10, 10] },
    ];
    for (const { list, grid, rect } of spec) {
        for (const c of list) {
            c.addEventListener('pointermove', (e) => {
                const r = c.getBoundingClientRect();
                const x = Math.min(grid - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * grid)));
                const y = Math.min(grid - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * grid)));
                drawPreviewWithField(rect(x, y));
            });
            c.addEventListener('pointerleave', () => drawPreviewWithField(null));
        }
    }
}

function drawPreviewWithField(rect) {
    const canvas = $('preview');
    drawHeat(canvas, state.input, 28, 28, { max: 1 });
    if (!rect) return;
    const ctx = canvas.getContext('2d');
    const px = canvas.width / 28;
    const [x, y, w, h] = rect;
    ctx.strokeStyle = '#86c3d6';
    ctx.lineWidth = 2;
    ctx.strokeRect(x * px + 1, y * px + 1, Math.min(w, 28 - x) * px - 2, Math.min(h, 28 - y) * px - 2);
}

// ---------- scan mode ----------
function buildScanPicker() {
    const box = $('scan-kernel-pick');
    box.textContent = '';
    for (let k = 0; k < 8; k++) {
        const c = document.createElement('canvas');
        c.width = 3;
        c.height = 3;
        c.style.width = '30px';
        c.style.height = '30px';
        c.className = 'pix' + (k === 0 ? ' is-active' : '');
        drawHeat(c, state.net.c1w.subarray(k * 9, k * 9 + 9), 3, 3, { signed: true });
        c.addEventListener('click', () => {
            state.scan.kernel = k;
            state.scan.pos = 0;
            box.querySelectorAll('canvas').forEach((x) => x.classList.toggle('is-active', x === c));
            drawScan();
        });
        box.append(c);
    }
}

function scanFrame() {
    if (state.scan.playing) {
        state.scan.pos += state.scan.speed;
        if (state.scan.pos >= 676) {
            state.scan.pos = 676;
            state.scan.playing = false;
            $('scan-play').textContent = 'scan';
        }
        drawScan();
    }
    requestAnimationFrame(scanFrame);
}

function drawScan() {
    if (!state.stages) return;
    const k = state.scan.kernel;
    const pos = Math.min(675, Math.floor(state.scan.pos));
    const oy = Math.floor(pos / 26);
    const ox = pos % 26;

    // input with the sliding window
    const inC = $('scan-input');
    drawHeat(inC, state.input, 28, 28, { max: 1 });
    const ctx = inC.getContext('2d');
    const px = inC.width / 28;
    ctx.strokeStyle = '#86c3d6';
    ctx.lineWidth = 2;
    ctx.strokeRect(ox * px + 0.5, oy * px + 0.5, 3 * px - 1, 3 * px - 1);

    // feature map painted up to pos (values are the real conv outputs)
    const mapC = $('scan-map');
    const conv = state.stages.conv1.subarray(k * 676, (k + 1) * 676);
    const revealed = new Float32Array(676);
    let peak = 1e-9;
    for (const v of conv) peak = Math.max(peak, Math.abs(v));
    for (let i = 0; i <= pos; i++) revealed[i] = conv[i];
    drawHeat(mapC, revealed, 26, 26, { signed: true, max: peak });

    // the arithmetic at this position
    const kw = state.net.c1w.subarray(k * 9, k * 9 + 9);
    const lines = [];
    let sum = state.net.c1b[k];
    const terms = [];
    for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
            const pv = state.input[(oy + ky) * 28 + ox + kx];
            const wv = kw[ky * 3 + kx];
            sum += pv * wv;
            terms.push(`${pv.toFixed(2)}×${wv >= 0 ? '+' : ''}${wv.toFixed(2)}`);
        }
    }
    lines.push(`position (${ox},${oy})   pixel × weight:`);
    lines.push(terms.slice(0, 3).join('  '));
    lines.push(terms.slice(3, 6).join('  '));
    lines.push(terms.slice(6, 9).join('  '));
    lines.push(`+ bias ${state.net.c1b[k].toFixed(2)}`);
    lines.push(`= <b>${sum.toFixed(3)}</b> → relu → <b>${Math.max(0, sum).toFixed(3)}</b>`);
    $('arith').innerHTML = lines.join('\n');
}

$('scan-play').addEventListener('click', () => {
    state.scan.playing = !state.scan.playing;
    if (state.scan.playing && state.scan.pos >= 676) state.scan.pos = 0;
    $('scan-play').textContent = state.scan.playing ? 'pause' : 'scan';
});
$('scan-step').addEventListener('click', () => {
    state.scan.playing = false;
    $('scan-play').textContent = 'scan';
    state.scan.pos = Math.min(676, Math.floor(state.scan.pos) + 1);
    drawScan();
});
$('scan-reset').addEventListener('click', () => {
    state.scan.pos = 0;
    state.scan.playing = false;
    $('scan-play').textContent = 'scan';
    drawScan();
});
$('scan-speed').addEventListener('input', (e) => {
    state.scan.speed = parseInt(e.target.value, 10);
});

// ---------- kernel playground ----------
const PRESETS = {
    edge: [0, -1, 0, -1, 4, -1, 0, -1, 0],
    'edge x': [-1, 0, 1, -2, 0, 2, -1, 0, 1],
    'edge y': [-1, -2, -1, 0, 0, 0, 1, 2, 1],
    sharpen: [0, -1, 0, -1, 5, -1, 0, -1, 0],
    blur: [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
    identity: [0, 0, 0, 0, 1, 0, 0, 0, 0],
};

function buildKernelEditor() {
    const box = $('kernel-edit');
    box.textContent = '';
    for (let i = 0; i < 9; i++) {
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.inputMode = 'decimal';
        inp.value = String(state.playKernel[i]);
        inp.addEventListener('change', () => {
            const v = parseFloat(inp.value);
            state.playKernel[i] = Number.isFinite(v) ? v : 0;
            inp.value = String(state.playKernel[i]);
            runPlayground();
        });
        box.append(inp);
    }
    const presets = $('kernel-presets');
    presets.textContent = '';
    for (const name of Object.keys(PRESETS)) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'mini-btn';
        b.textContent = name;
        b.addEventListener('click', () => {
            state.playKernel = PRESETS[name].slice();
            [...box.children].forEach((inp, i) => { inp.value = String(state.playKernel[i]); });
            runPlayground();
        });
        presets.append(b);
    }
}

function runPlayground() {
    const out = conv2dForward(state.input, Float32Array.from(state.playKernel), null, 1, 28, 28, 1, 3, 3);
    drawHeat($('play-out'), out, 26, 26, { signed: true });
}
