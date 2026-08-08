import { fetchWeights } from '../engine/weights.js';
import { fetchDataset, preprocessDrawing } from '../engine/mnist.js';
import { MLP } from '../engine/mlp.js';
import { makeSchedule, forwardNoise, tEmbed, createSampler } from '../engine/diffusion.js';
import { mulberry32, fillGaussian } from '../engine/rng.js';
import { drawSignedImage, drawHeat } from '../ui/heat.js';
import { DrawBox } from '../ui/drawbox.js';

const $ = (id) => document.getElementById(id);

const CLASS_COLORS = [
    '#6a9955', '#c25b4e', '#86c3d6', '#c2a24e', '#9d7bd8',
    '#4ec2a4', '#d88ab0', '#8fb54e', '#d8873b', '#7b95d8',
];

const state = {
    meta: null,
    denoiser: null,
    ce: null,
    alphaBar: null,
    enc: null,
    dec: null,
    test: null,
    // forward tab
    fwdX0: new Float32Array(784),
    fwdEps: fillGaussian(new Float32Array(784), mulberry32(1234)),
    // reverse tab
    classIdx: 3,
    sampler: null,
    playing: false,
    lastStep: 0,
    curFrame: -1,
    // latent tab
    latentPoints: null,
    z: [0, 0],
    zA: null,
    zB: null,
    latDraw: null,
};

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
            text.textContent = `fetching models… ${pct}%`;
        };
        const [dw, aw, test] = await Promise.all([
            fetchWeights('./data/weights/diffusion-ref.bin', track(0)),
            fetchWeights('./data/weights/ae-ref.bin', track(1)),
            fetchDataset('./data/mnist-test.bin', track(2)),
        ]);
        state.meta = dw.meta;
        state.denoiser = new MLP(dw.meta.sizes, { task: 'regress', hidden: dw.meta.hidden }, mulberry32(1))
            .loadTensors(dw.tensors);
        state.ce = dw.tensors.ce.data;
        state.alphaBar = makeSchedule(dw.meta.T);
        state.enc = new MLP(aw.meta.encSizes, { task: 'regress' }, mulberry32(2)).loadTensors(aw.tensors, 'enc_');
        state.dec = new MLP(aw.meta.decSizes, { task: 'regress' }, mulberry32(3)).loadTensors(aw.tensors, 'dec_');
        state.test = test;
    } catch (err) {
        text.textContent = `failed to load: ${err.message}`;
        throw err;
    }
    $('loader').classList.add('done');
    $('net-status').textContent =
        `denoiser 1.1M params · T=${state.meta.T} · samples verified by the CNN at ${(state.meta.cnnAgreement * 100).toFixed(0)}%`;

    buildTabs();
    buildForward();
    buildReverse();
    buildLatent();
    requestAnimationFrame(loop);
})();

// ---------- tabs ----------
function buildTabs() {
    const tabs = document.querySelectorAll('.tab-row .seg-btn');
    tabs.forEach((b) => {
        b.addEventListener('click', () => {
            tabs.forEach((x) => x.classList.toggle('is-active', x === b));
            $('tab-diffusion').hidden = b.dataset.tab !== 'diffusion';
            $('tab-latent').hidden = b.dataset.tab !== 'latent';
        });
    });
}

// ---------- the denoiser wrappers ----------
function predictX0(xt, t, cls) {
    const meta = state.meta;
    const input = new Float32Array(meta.sizes[0]);
    input.set(xt, 0);
    input.set(tEmbed(t, meta.tDim), 784);
    input.set(state.ce.subarray(cls * meta.cDim, (cls + 1) * meta.cDim), 784 + meta.tDim);
    const out = state.denoiser.predict(input);
    const x0 = new Float32Array(784);
    for (let i = 0; i < 784; i++) x0[i] = Math.min(1, Math.max(-1, out[i]));
    return x0;
}

function predictEps(xt, t, cls) {
    const x0 = predictX0(xt, t, cls);
    const sa = Math.sqrt(state.alphaBar[t]);
    const sb = Math.sqrt(1 - state.alphaBar[t]);
    const eps = new Float32Array(784);
    for (let i = 0; i < 784; i++) eps[i] = (xt[i] - sa * x0[i]) / sb;
    return eps;
}

// ---------- forward tab ----------
function buildForward() {
    const strip = $('fwd-samples');
    const rand = mulberry32(77);
    const used = new Set();
    for (let d = 0; d < 10; d++) {
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
        c.style.width = '32px';
        c.style.height = '32px';
        c.className = 'pix';
        const img = state.test.image(idx);
        drawHeat(c, img, 28, 28, { max: 1 });
        c.addEventListener('click', () => {
            for (let p = 0; p < 784; p++) state.fwdX0[p] = img[p] * 2 - 1;
            drawForward();
        });
        strip.append(c);
    }
    // default: first sample
    const first = state.test.image(0);
    for (let p = 0; p < 784; p++) state.fwdX0[p] = first[p] * 2 - 1;

    $('fwd-t').addEventListener('input', drawForward);
    drawSchedule();
    drawForward();
}

function drawForward() {
    const t = parseInt($('fwd-t').value, 10);
    $('fwd-t-out').textContent = String(t);
    const abar = state.alphaBar[t];
    const xt = forwardNoise(state.fwdX0, state.fwdEps, abar);
    drawSignedImage($('fwd-x0'), state.fwdX0, 28, 28);
    drawSignedImage($('fwd-xt'), xt, 28, 28);
    $('fwd-math').innerHTML =
        `x<sub>t</sub> = √ᾱ·x₀ + √(1−ᾱ)·ε\n` +
        `   = <b>${Math.sqrt(abar).toFixed(3)}</b>·x₀ + <b>${Math.sqrt(1 - abar).toFixed(3)}</b>·ε`;
    drawSchedule(t);
}

function drawSchedule(tMark = 0) {
    const canvas = $('schedule');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const pad = 24;
    ctx.strokeStyle = '#24292d';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    ctx.lineTo(w - pad, h - pad);
    ctx.stroke();

    ctx.strokeStyle = '#86c3d6';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let t = 0; t <= state.meta.T; t++) {
        const x = pad + (t / state.meta.T) * (w - 2 * pad);
        const y = h - pad - state.alphaBar[t] * (h - 2 * pad);
        if (t === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const mx = pad + (tMark / state.meta.T) * (w - 2 * pad);
    ctx.fillStyle = '#eeebe3';
    ctx.beginPath();
    ctx.arc(mx, h - pad - state.alphaBar[tMark] * (h - 2 * pad), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#5f5c55';
    ctx.font = '17px "Plex Mono", ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('ᾱ', 6, 20);
    ctx.textAlign = 'right';
    ctx.fillText('t →', w - 6, h - 6);
}

// ---------- reverse tab ----------
function buildReverse() {
    const box = $('class-pick');
    for (let d = 0; d < 10; d++) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'seg-btn' + (d === state.classIdx ? ' is-active' : '');
        b.textContent = String(d);
        b.addEventListener('click', () => {
            state.classIdx = d;
            box.querySelectorAll('.seg-btn').forEach((x) => x.classList.toggle('is-active', x === b));
        });
        box.append(b);
    }
    $('guide').addEventListener('input', (e) => {
        $('guide-out').textContent = parseFloat(e.target.value).toFixed(1);
    });
    $('seed').addEventListener('input', (e) => {
        $('seed-out').textContent = e.target.value;
    });
    $('generate').addEventListener('click', () => {
        startSampler();
        state.playing = true;
        $('generate').textContent = 'restart';
    });
    $('rev-step').addEventListener('click', () => {
        if (!state.sampler || state.sampler.done) startSampler();
        state.playing = false;
        doStep();
    });
    // idle preview: pure noise
    const noise = fillGaussian(new Float32Array(784), mulberry32(7));
    drawSignedImage($('rev-xt'), noise, 28, 28);
}

function startSampler() {
    state.sampler = createSampler(predictEps, {
        T: state.meta.T,
        steps: parseInt($('rev-steps').value, 10),
        classIdx: state.classIdx,
        guidance: parseFloat($('guide').value),
        seed: parseInt($('seed').value, 10),
    });
    state.curFrame = -1;
    $('film').textContent = '';
}

function doStep() {
    const rec = state.sampler.step();
    if (!rec) {
        state.playing = false;
        $('generate').textContent = 'generate';
        return;
    }
    renderFrame(state.sampler.history.length - 1);
    // film thumbnail
    const c = document.createElement('canvas');
    c.width = 28;
    c.height = 28;
    drawSignedImage(c, rec.xt, 28, 28);
    const frameIdx = state.sampler.history.length - 1;
    c.addEventListener('pointerenter', () => renderFrame(frameIdx));
    $('film').append(c);
    [...$('film').children].forEach((x, i) => x.classList.toggle('is-cur', i === frameIdx));
}

function renderFrame(i) {
    const rec = state.sampler.history[i];
    if (!rec) return;
    state.curFrame = i;
    drawSignedImage($('rev-xt'), rec.xt, 28, 28);
    drawSignedImage($('rev-x0'), rec.x0Hat, 28, 28);
    drawSignedImage($('rev-eps'), rec.epsHat, 28, 28);
    $('rev-cap').innerHTML = `x<sub>t</sub> — step ${i + 1}/${state.sampler.history.length + (state.sampler.done ? 0 : '…')} · t=${rec.tPrev}`;
    [...$('film').children].forEach((x, k) => x.classList.toggle('is-cur', k === i));
}

// ---------- latent tab ----------
function buildLatent() {
    state.latDraw = new DrawBox($('lat-draw'), {
        onchange(value) {
            const input = preprocessDrawing(value);
            const z = state.enc.predict(input);
            state.z = [z[0], z[1]];
            $('lat-z').innerHTML = `z = ( <b>${z[0].toFixed(2)}</b> , <b>${z[1].toFixed(2)}</b> )`;
            decodeAt(z[0], z[1]);
            drawLatentMap();
        },
    });
    $('lat-clear').addEventListener('click', () => state.latDraw.clear());
    $('lat-undo').addEventListener('click', () => state.latDraw.undo());

    const strip = $('lat-samples');
    const fillStrip = () => {
        strip.textContent = '';
        const rand = mulberry32(Date.now() & 0xffff);
        for (let d = 0; d < 10; d++) {
            let idx = -1;
            for (let tries = 0; tries < 200; tries++) {
                const i = Math.floor(rand() * state.test.count);
                if (state.test.labels[i] === d) { idx = i; break; }
            }
            if (idx < 0) continue;
            const c = document.createElement('canvas');
            c.width = 28;
            c.height = 28;
            c.style.width = '26px';
            c.style.height = '26px';
            c.className = 'pix';
            const img = state.test.image(idx);
            drawHeat(c, img, 28, 28, { max: 1 });
            c.addEventListener('click', () => state.latDraw.setValue(img));
            strip.append(c);
        }
    };
    $('lat-shuffle').addEventListener('click', fillStrip);
    fillStrip();

    // encode the whole shipped test set (the scatter) — chunked to keep UI alive
    state.latentPoints = [];
    const x = new Float32Array(784);
    let i = 0;
    const chunk = () => {
        const until = Math.min(state.test.count, i + 100);
        for (; i < until; i++) {
            const z = state.enc.predict(state.test.image(i, x));
            state.latentPoints.push([z[0], z[1], state.test.labels[i]]);
        }
        drawLatentMap();
        if (i < state.test.count) setTimeout(chunk, 0);
    };
    chunk();

    const map = $('lat-map');
    let dragging = false;
    const pick = (e) => {
        const r = map.getBoundingClientRect();
        const bounds = latentBounds();
        const px = ((e.clientX - r.left) / r.width) * map.width;
        const py = ((e.clientY - r.top) / r.height) * map.height;
        const z0 = bounds.x0 + (px / map.width) * (bounds.x1 - bounds.x0);
        const z1 = bounds.y1 - (py / map.height) * (bounds.y1 - bounds.y0);
        state.z = [z0, z1];
        decodeAt(z0, z1);
        drawLatentMap();
    };
    map.addEventListener('pointerdown', (e) => {
        dragging = true;
        try { map.setPointerCapture(e.pointerId); } catch { /* synthetic pointers */ }
        pick(e);
    });
    map.addEventListener('pointermove', (e) => { if (dragging) pick(e); });
    map.addEventListener('pointerup', () => { dragging = false; });

    $('set-a').addEventListener('click', () => { state.zA = state.z.slice(); lerpMove(); });
    $('set-b').addEventListener('click', () => { state.zB = state.z.slice(); lerpMove(); });
    $('lerp').addEventListener('input', lerpMove);

    decodeAt(0, 0);
}

function lerpMove() {
    if (!state.zA || !state.zB) return;
    const f = $('lerp').value / 100;
    const z0 = state.zA[0] + f * (state.zB[0] - state.zA[0]);
    const z1 = state.zA[1] + f * (state.zB[1] - state.zA[1]);
    state.z = [z0, z1];
    decodeAt(z0, z1);
    drawLatentMap();
}

function decodeAt(z0, z1) {
    const out = state.dec.predict(Float32Array.from([z0, z1]));
    const img = new Float32Array(784);
    for (let i = 0; i < 784; i++) img[i] = Math.min(1, Math.max(0, out[i]));
    drawHeat($('lat-out'), img, 28, 28, { max: 1 });
    $('dec-z').innerHTML = `decoder( <b>${z0.toFixed(2)}</b> , <b>${z1.toFixed(2)}</b> )`;
}

let latentBoundsCache = null;
function latentBounds() {
    if (latentBoundsCache && latentBoundsCache.n === state.latentPoints.length) return latentBoundsCache;
    let x0 = -1, x1 = 1, y0 = -1, y1 = 1;
    for (const [x, y] of state.latentPoints) {
        x0 = Math.min(x0, x); x1 = Math.max(x1, x);
        y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    const padX = (x1 - x0) * 0.06;
    const padY = (y1 - y0) * 0.06;
    latentBoundsCache = { x0: x0 - padX, x1: x1 + padX, y0: y0 - padY, y1: y1 + padY, n: state.latentPoints.length };
    return latentBoundsCache;
}

function drawLatentMap() {
    const canvas = $('lat-map');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const b = latentBounds();
    const sx = (x) => ((x - b.x0) / (b.x1 - b.x0)) * w;
    const sy = (y) => h - ((y - b.y0) / (b.y1 - b.y0)) * h;

    for (const [x, y, label] of state.latentPoints) {
        ctx.fillStyle = CLASS_COLORS[label];
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        ctx.arc(sx(x), sy(y), 3, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.globalAlpha = 1;

    // legend
    ctx.font = '16px "Plex Mono", ui-monospace, monospace';
    for (let d = 0; d < 10; d++) {
        ctx.fillStyle = CLASS_COLORS[d];
        ctx.fillText(String(d), 10 + d * 20, 20);
    }

    // interpolation line
    if (state.zA && state.zB) {
        ctx.strokeStyle = 'rgba(238,235,227,0.4)';
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(sx(state.zA[0]), sy(state.zA[1]));
        ctx.lineTo(sx(state.zB[0]), sy(state.zB[1]));
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // the current point
    ctx.strokeStyle = '#eeebe3';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(sx(state.z[0]), sy(state.z[1]), 8, 0, Math.PI * 2);
    ctx.stroke();
}

// ---------- loop ----------
function loop(now) {
    if (state.playing && state.sampler && now - state.lastStep > 90) {
        state.lastStep = now;
        doStep();
    }
    requestAnimationFrame(loop);
}
