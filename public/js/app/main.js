import { mulberry32 } from '../nn/rng.js';
import { datasets } from '../nn/data.js';
import { Network } from '../nn/network.js';
import { drawDiagram } from './diagram.js';
import { drawBoundary } from './boundary.js';
import { drawLoss } from './losschart.js';

const N_POINTS = 240;
const BATCH = 24;
const MAX_LAYERS = 5;
const MAX_NEURONS = 8;

const state = {
    datasetName: 'circles',
    noise: 0.1,
    hidden: [4, 4],
    activation: 'tanh',
    lr: 0.3,
    dataSeed: 1337,
    netSeed: 1,
    playing: false,
    epoch: 0,
    lossHistory: [],
    data: [],
    net: null,
};

const el = (id) => document.getElementById(id);
const diagramCanvas = el('diagram');
const boundaryCanvas = el('boundary');
const lossCanvas = el('losschart');
let dirty = true;

function rebuildData() {
    state.data = datasets[state.datasetName](N_POINTS, state.noise, mulberry32(state.dataSeed));
}

function rebuildNet() {
    state.net = new Network([2, ...state.hidden, 1], state.activation, mulberry32(state.netSeed));
    state.epoch = 0;
    state.lossHistory = [];
    el('shape-note').textContent = state.net.sizes.join(' → ');
    dirty = true;
}

// One pass over the data in minibatches; the data order is already random.
function trainEpoch() {
    let sum = 0;
    let batches = 0;
    for (let i = 0; i < state.data.length; i += BATCH) {
        sum += state.net.trainBatch(state.data.slice(i, i + BATCH), state.lr);
        batches++;
    }
    state.epoch++;
    state.lossHistory.push(sum / batches);
    dirty = true;
}

function drawAll() {
    drawDiagram(state.net, diagramCanvas);
    drawBoundary(state.net, state.data, boundaryCanvas);
    drawLoss(state.lossHistory, lossCanvas);
    el('stat-epoch').textContent = String(state.epoch);
    const last = state.lossHistory[state.lossHistory.length - 1];
    el('stat-loss').textContent = last === undefined ? '–' : last.toFixed(3);
    el('stat-acc').textContent = `${Math.round(state.net.accuracy(state.data) * 100)}%`;
    dirty = false;
}

function frame() {
    if (state.playing) trainEpoch();
    if (dirty) drawAll();
    requestAnimationFrame(frame);
}

function setPlaying(on) {
    state.playing = on;
    el('play').textContent = on ? 'pause' : 'train';
}

// ---- layer editor ----

function renderLayers() {
    const box = el('layers');
    box.textContent = '';
    state.hidden.forEach((count, idx) => {
        const chip = document.createElement('div');
        chip.className = 'layer-chip';

        const label = document.createElement('span');
        label.className = 'lc-label';
        label.textContent = `hidden ${idx + 1}`;

        const minus = document.createElement('button');
        minus.type = 'button';
        minus.className = 'lc-btn';
        minus.textContent = '−';
        minus.setAttribute('aria-label', `fewer neurons in hidden layer ${idx + 1}`);
        minus.addEventListener('click', () => {
            if (state.hidden[idx] > 1) {
                state.hidden[idx]--;
                renderLayers();
                rebuildNet();
            }
        });

        const value = document.createElement('span');
        value.className = 'lc-count';
        value.textContent = String(count);

        const plus = document.createElement('button');
        plus.type = 'button';
        plus.className = 'lc-btn';
        plus.textContent = '+';
        plus.setAttribute('aria-label', `more neurons in hidden layer ${idx + 1}`);
        plus.addEventListener('click', () => {
            if (state.hidden[idx] < MAX_NEURONS) {
                state.hidden[idx]++;
                renderLayers();
                rebuildNet();
            }
        });

        chip.append(label, minus, value, plus);
        box.append(chip);
    });
}

// ---- controls ----

el('dataset-picker').addEventListener('click', (e) => {
    const btn = e.target.closest('.seg-btn');
    if (!btn) return;
    state.datasetName = btn.dataset.set;
    document.querySelectorAll('.seg-btn').forEach((b) => {
        b.classList.toggle('is-active', b === btn);
    });
    setPlaying(false);
    rebuildData();
    rebuildNet();
});

el('noise').addEventListener('input', (e) => {
    state.noise = parseFloat(e.target.value);
    el('noise-out').textContent = state.noise.toFixed(2);
    setPlaying(false);
    rebuildData();
    rebuildNet();
});

el('activation').addEventListener('change', (e) => {
    state.activation = e.target.value;
    setPlaying(false);
    rebuildNet();
});

el('lr').addEventListener('change', (e) => {
    state.lr = parseFloat(e.target.value);
});

el('add-layer').addEventListener('click', () => {
    if (state.hidden.length < MAX_LAYERS) {
        state.hidden.push(4);
        renderLayers();
        rebuildNet();
    }
});

el('remove-layer').addEventListener('click', () => {
    if (state.hidden.length > 1) {
        state.hidden.pop();
        renderLayers();
        rebuildNet();
    }
});

el('play').addEventListener('click', () => setPlaying(!state.playing));

el('step').addEventListener('click', () => {
    setPlaying(false);
    trainEpoch();
});

el('reset').addEventListener('click', () => {
    setPlaying(false);
    state.netSeed++; // fresh weight init each reset
    rebuildNet();
});

window.addEventListener('resize', () => { dirty = true; });

// ---- boot ----

rebuildData();
renderLayers();
rebuildNet();
requestAnimationFrame(frame);
