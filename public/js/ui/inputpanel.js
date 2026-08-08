import { DrawBox } from './drawbox.js';
import { drawHeat } from './heat.js';
import { mulberry32 } from '../engine/rng.js';
import { preprocessDrawing } from '../engine/mnist.js';

// The draw-a-digit input rail shared by the CNN and diffusion pages:
// draw box + clear/undo + real-sample strip + preprocessed preview.
// onInput receives the preprocessed Float32Array(784).
export function initInputPanel({ drawId, clearId, undoId, samplesId, shuffleId, previewId, testSet, onInput }) {
    const previewEl = previewId ? document.getElementById(previewId) : null;
    let current = new Float32Array(784);

    const box = new DrawBox(document.getElementById(drawId), {
        onchange(value) {
            current = preprocessDrawing(value);
            if (previewEl) drawHeat(previewEl, current, 28, 28, { max: 1 });
            onInput(current, box);
        },
    });

    document.getElementById(clearId).addEventListener('click', () => box.clear());
    document.getElementById(undoId).addEventListener('click', () => box.undo());

    function fillSamples() {
        const strip = document.getElementById(samplesId);
        strip.textContent = '';
        const rand = mulberry32(Date.now() & 0xffff);
        const used = new Set();
        for (let d = 0; d < 10; d++) {
            let idx = -1;
            for (let tries = 0; tries < 200; tries++) {
                const i = Math.floor(rand() * testSet.count);
                if (testSet.labels[i] === d && !used.has(i)) { idx = i; break; }
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
            const img = testSet.image(idx);
            drawHeat(c, img, 28, 28, { max: 1 });
            c.addEventListener('click', () => box.setValue(img));
            strip.append(c);
        }
    }
    document.getElementById(shuffleId).addEventListener('click', fillSamples);
    fillSamples();

    return {
        box,
        get value() { return current; },
    };
}
