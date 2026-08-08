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
    preRelu: false,
    normalize: false,
    inspect: null, // { stage, idx } of the feature map open in the inspector
};

const bars = new Bars($('bars'), [...Array(10).keys()].map(String));
const mapCanvases = { maps1: [], pool1: [], maps2: [], pool2: [] };

// grid size and headline for each clickable feature-map row
const STAGE_INFO = {
    maps1: { name: 'conv1', grid: 26, count: 8 },
    pool1: { name: 'pool1', grid: 13, count: 8 },
    maps2: { name: 'conv2', grid: 11, count: 16 },
    pool2: { name: 'pool2', grid: 5, count: 16 },
};

let panel = null;

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

    panel = initInputPanel({
        drawId: 'draw', clearId: 'clear', undoId: 'undo',
        samplesId: 'samples', shuffleId: 'shuffle-samples', previewId: 'preview',
        testSet: test,
        onInput: (input) => {
            state.input = input;
            runForward();
        },
    });
    buildBrushPicker();

    runForward(); // draw every stage once, even before the first stroke

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
    wireMapClicks();
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

const BRUSHES = [['thin', 0.75], ['medium', 1.15], ['thick', 1.7]];

function buildBrushPicker() {
    const box = $('brush-pick');
    box.textContent = '';
    BRUSHES.forEach(([name, sigma], i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'seg-btn' + (i === 1 ? ' is-active' : '');
        b.textContent = name;
        b.title = `${name} strokes`;
        b.addEventListener('click', () => {
            panel.box.setBrush(sigma);
            box.querySelectorAll('.seg-btn').forEach((x) => x.classList.toggle('is-active', x === b));
        });
        box.append(b);
    });
}

// peak magnitude of the values currently shown for a conv row, so the whole row
// shares one scale and maps stay comparable to each other
function rowPeak(data) {
    let peak = 1e-9;
    for (const v of data) peak = Math.max(peak, Math.abs(v));
    return peak;
}

// the tensor a given row is showing right now — pre-relu when the toggle is on
function stageData(stage) {
    const s = state.stages;
    if (stage === 'maps1') return state.preRelu ? s.conv1 : s.relu1;
    if (stage === 'maps2') return state.preRelu ? s.conv2 : s.relu2;
    if (stage === 'pool1') return s.pool1.out;
    return s.pool2.out;
}

// one scale per row, so maps in a row stay comparable; the pools always keep the
// post-relu scale so the toggle never changes how they look
function rowScale(stage) {
    if (stage === 'pool1') return rowPeak(state.stages.relu1);
    if (stage === 'pool2') return rowPeak(state.stages.relu2);
    return rowPeak(stageData(stage));
}

function drawMaps() {
    const s = state.stages;
    if (!s) return;
    const signed = state.preRelu;

    const conv1Data = stageData('maps1');
    const peak1 = rowScale('maps1');
    const poolPeak1 = rowScale('pool1');
    for (let k = 0; k < 8; k++) {
        drawHeat(mapCanvases.maps1[k], conv1Data.subarray(k * 676, (k + 1) * 676), 26, 26, { signed, max: peak1 });
        drawHeat(mapCanvases.pool1[k], s.pool1.out.subarray(k * 169, (k + 1) * 169), 13, 13, { max: poolPeak1 });
    }

    const conv2Data = stageData('maps2');
    const peak2 = rowScale('maps2');
    const poolPeak2 = rowScale('pool2');
    for (let k = 0; k < 16; k++) {
        drawHeat(mapCanvases.maps2[k], conv2Data.subarray(k * 121, (k + 1) * 121), 11, 11, { signed, max: peak2 });
        drawHeat(mapCanvases.pool2[k], s.pool2.out.subarray(k * 25, (k + 1) * 25), 5, 5, { max: poolPeak2 });
    }
    drawHeat($('flat'), s.flat, 400, 1, { max: poolPeak2 });

    $('cap1').textContent = state.preRelu
        ? '→ 8 feature maps, before relu — negative responses in red'
        : '→ 8 feature maps (after relu)';
    $('cap2').textContent = state.preRelu
        ? 'conv 2 — 16 kernels over all 8 maps, before relu — negative responses in red'
        : 'conv 2 — 16 kernels over all 8 maps → relu';
}

function runForward() {
    if (!state.net) return;
    const s = state.net.forward(state.input);
    state.stages = s;

    drawMaps();

    let best = 0;
    for (let o = 1; o < 10; o++) if (s.probs[o] > s.probs[best]) best = o;
    bars.update(Array.from(s.probs), best);
    $('pred').textContent = state.input.some((v) => v > 0.02) ? String(best) : '–';

    state.scan.pos = 0; // restart the scan against fresh input
    drawScan();
    drawInspector();
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

// ---------- feature-map inspector ----------
function wireMapClicks() {
    for (const stage of Object.keys(STAGE_INFO)) {
        mapCanvases[stage].forEach((c, idx) => {
            c.title = `${STAGE_INFO[stage].name} map ${idx + 1} — click to inspect`;
            c.addEventListener('click', () => {
                state.inspect = { stage, idx };
                drawInspector();
            });
        });
    }
}

function drawInspector() {
    const pick = state.inspect;
    if (!pick || !state.stages) {
        $('inspector').hidden = true;
        return;
    }
    const info = STAGE_INFO[pick.stage];
    const grid = info.grid;
    const len = grid * grid;
    const data = stageData(pick.stage).subarray(pick.idx * len, (pick.idx + 1) * len);
    const isConv = pick.stage === 'maps1' || pick.stage === 'maps2';

    const canvas = $('insp-map');
    canvas.width = grid;
    canvas.height = grid;
    drawHeat(canvas, data, grid, grid, { signed: state.preRelu && isConv, max: rowScale(pick.stage) });

    $('insp-cap').textContent = `${info.name} · map ${pick.idx + 1} of ${info.count} · ${grid}×${grid}`;

    const wrap = $('insp-kernel-wrap');
    wrap.hidden = pick.stage !== 'maps1';
    if (pick.stage === 'maps1') {
        drawHeat($('insp-kernel'), state.net.c1w.subarray(pick.idx * 9, pick.idx * 9 + 9), 3, 3, { signed: true });
    }

    let lo = Infinity;
    let hi = -Infinity;
    for (const v of data) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    const range = `values run ${lo.toFixed(2)} to ${hi.toFixed(2)}`;
    let why;
    if (pick.stage === 'maps1') {
        why = 'this map is the kernel beside it slid over all 26×26 positions of your drawing.';
    } else if (pick.stage === 'maps2') {
        why = 'each conv-2 map mixes all 8 pooled maps at once, so there is no single kernel to show.';
    } else {
        why = 'pooling kept the largest value in each 2×2 block of the map above.';
    }
    $('insp-note').textContent = `${why} ${range}, drawn on the same scale as its row.`;

    $('inspector').hidden = false;
}

$('insp-close').addEventListener('click', () => {
    state.inspect = null;
    $('inspector').hidden = true;
});

$('pre-relu').addEventListener('change', (e) => {
    state.preRelu = e.target.checked;
    drawMaps();
    drawInspector();
});

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

    $('scan-pos').value = String(pos);
    $('scan-pos-v').textContent = `${pos} / 675`;

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
$('scan-pos').addEventListener('input', (e) => {
    state.scan.playing = false;
    $('scan-play').textContent = 'scan';
    state.scan.pos = parseInt(e.target.value, 10);
    drawScan();
});

// ---------- kernel playground ----------
const PRESETS = {
    edge: [0, -1, 0, -1, 4, -1, 0, -1, 0],
    'edge x': [-1, 0, 1, -2, 0, 2, -1, 0, 1],
    'edge y': [-1, -2, -1, 0, 0, 0, 1, 2, 1],
    sharpen: [0, -1, 0, -1, 5, -1, 0, -1, 0],
    blur: [0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
    emboss: [-2, -1, 0, -1, 1, 1, 0, 1, 2],
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

// dividing by the sum of |entries| keeps the response in the same range as the
// input instead of letting big weights blow the scale out
function effectiveKernel() {
    const k = Float32Array.from(state.playKernel);
    if (!state.normalize) return k;
    let mass = 0;
    for (const v of k) mass += Math.abs(v);
    if (mass === 0) return k;
    for (let i = 0; i < k.length; i++) k[i] /= mass;
    return k;
}

// the picture is auto-scaled to its own peak, so normalizing never changes what
// it looks like — only the size of the numbers. Show that number.
function updateNormNote(peak) {
    const strongest = `strongest response ${peak.toFixed(3)}`;
    if (!state.normalize) {
        $('norm-note').textContent = `raw values are used as typed — ${strongest}.`;
        return;
    }
    let mass = 0;
    for (const v of state.playKernel) mass += Math.abs(v);
    $('norm-note').textContent = mass === 0
        ? `all-zero kernel — nothing to divide by, so the raw values are used. ${strongest}.`
        : `each value is divided by ${mass.toFixed(2)}, the sum of |values| — ${strongest}, which keeps the response in the same range as the input. The inputs still show what you typed, and the picture keeps its shape: only the scale moved.`;
}

$('kernel-norm').addEventListener('change', (e) => {
    state.normalize = e.target.checked;
    runPlayground();
});

function runPlayground() {
    const out = conv2dForward(state.input, effectiveKernel(), null, 1, 28, 28, 1, 3, 3);
    drawHeat($('play-out'), out, 26, 26, { signed: true });
    let peak = 0;
    for (const v of out) peak = Math.max(peak, Math.abs(v));
    updateNormNote(peak);
}
