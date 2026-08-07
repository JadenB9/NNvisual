import { affineForward, affineBackward, relu, reluBackward, tanhAct, sigmoid, softmaxRows } from './tensor.js';

// Dense network on flat Float32Arrays. One class powers the classifier
// playground, the autoencoder, and the diffusion denoiser:
//   task "classify": softmax output + cross-entropy on integer labels
//   task "regress":  linear output + mean-squared error on float targets
//
// Weights: w[l] is Float32Array(nOut*nIn), row per output unit; b[l] biases.
export class MLP {
    constructor(sizes, { hidden = 'relu', task = 'classify' } = {}, rand = Math.random) {
        this.sizes = sizes.slice();
        this.hidden = hidden;
        this.task = task;
        this.weights = [];
        this.biases = [];
        this.vw = []; // momentum buffers
        this.vb = [];
        for (let l = 0; l < sizes.length - 1; l++) {
            const nIn = sizes[l];
            const nOut = sizes[l + 1];
            const w = new Float32Array(nOut * nIn);
            const scale = hidden === 'relu' ? Math.sqrt(2 / nIn) : Math.sqrt(1 / nIn);
            for (let i = 0; i < w.length; i++) w[i] = (rand() * 2 - 1) * scale * Math.sqrt(3);
            this.weights.push(w);
            this.biases.push(new Float32Array(nOut));
            this.vw.push(new Float32Array(nOut * nIn));
            this.vb.push(new Float32Array(nOut));
        }
    }

    get depth() { return this.weights.length; }

    activate(z, out) {
        if (this.hidden === 'relu') return relu(z, out);
        if (this.hidden === 'tanh') return tanhAct(z, out);
        return sigmoid(z, out);
    }

    activateBackward(a, dy, dx) {
        if (this.hidden === 'relu') return reluBackward(a, dy, dx);
        for (let i = 0; i < a.length; i++) {
            dx[i] = this.hidden === 'tanh' ? dy[i] * (1 - a[i] * a[i]) : dy[i] * a[i] * (1 - a[i]);
        }
        return dx;
    }

    // Batched forward. Returns { acts, zs, out }:
    //   acts[0] = input batch, acts[l+1] = post-activation of layer l
    //   out = probabilities (classify) or raw output (regress)
    forwardBatch(x, batch) {
        const acts = [x];
        const zs = [];
        let cur = x;
        for (let l = 0; l < this.depth; l++) {
            const nIn = this.sizes[l];
            const nOut = this.sizes[l + 1];
            const z = affineForward(cur, this.weights[l], this.biases[l], batch, nIn, nOut);
            zs.push(z);
            const last = l === this.depth - 1;
            const a = last ? z : this.activate(z);
            acts.push(a);
            cur = a;
        }
        const logits = acts[acts.length - 1];
        const out = this.task === 'classify'
            ? softmaxRows(logits, batch, this.sizes[this.sizes.length - 1])
            : logits;
        return { acts, zs, out };
    }

    // Single-sample forward that keeps every layer for the visualizations.
    forwardSample(x) {
        const { acts, out } = this.forwardBatch(x, 1);
        return { acts, out };
    }

    predict(x) {
        return this.forwardBatch(x, 1).out;
    }

    // One SGD(+momentum) step. labels: Int/Uint array (classify) or
    // Float32Array targets (regress). Returns mean batch loss.
    // Also stashes the last per-layer gradient arrays on this.lastGrads for
    // the backprop visualization — they are the real gradients just applied.
    trainBatch(x, labels, batch, lr, momentum = 0.9) {
        const { acts, out } = this.forwardBatch(x, batch);
        const nOut = this.sizes[this.sizes.length - 1];

        // output delta and loss
        let loss = 0;
        const delta = new Float32Array(batch * nOut);
        if (this.task === 'classify') {
            for (let b = 0; b < batch; b++) {
                const off = b * nOut;
                const y = labels[b];
                const p = Math.max(out[off + y], 1e-12);
                loss -= Math.log(p);
                for (let o = 0; o < nOut; o++) {
                    delta[off + o] = (out[off + o] - (o === y ? 1 : 0)) / batch;
                }
            }
            loss /= batch;
        } else {
            for (let i = 0; i < batch * nOut; i++) {
                const d = out[i] - labels[i];
                loss += d * d;
                delta[i] = (2 * d) / (batch * nOut);
            }
            loss /= batch * nOut;
        }

        // backward
        const gw = [];
        const gb = [];
        let dy = delta;
        for (let l = this.depth - 1; l >= 0; l--) {
            const nIn = this.sizes[l];
            const nO = this.sizes[l + 1];
            const dw = new Float32Array(nO * nIn);
            const db = new Float32Array(nO);
            const dx = l > 0 ? new Float32Array(batch * nIn) : null;
            affineBackward(acts[l], this.weights[l], dy, batch, nIn, nO, dw, db, dx);
            gw[l] = dw;
            gb[l] = db;
            if (l > 0) {
                this.activateBackward(acts[l], dx, dx);
                dy = dx;
            }
        }

        // update
        for (let l = 0; l < this.depth; l++) {
            const w = this.weights[l];
            const b = this.biases[l];
            const vw = this.vw[l];
            const vb = this.vb[l];
            for (let i = 0; i < w.length; i++) {
                vw[i] = momentum * vw[i] - lr * gw[l][i];
                w[i] += vw[i];
            }
            for (let i = 0; i < b.length; i++) {
                vb[i] = momentum * vb[i] - lr * gb[l][i];
                b[i] += vb[i];
            }
        }
        this.lastGrads = { gw, gb };
        return loss;
    }

    // Mean loss over a dataset slice (no update).
    evalLoss(x, labels, count) {
        const { out } = this.forwardBatch(x, count);
        const nOut = this.sizes[this.sizes.length - 1];
        let loss = 0;
        if (this.task === 'classify') {
            for (let b = 0; b < count; b++) {
                loss -= Math.log(Math.max(out[b * nOut + labels[b]], 1e-12));
            }
            return loss / count;
        }
        for (let i = 0; i < count * nOut; i++) {
            const d = out[i] - labels[i];
            loss += d * d;
        }
        return loss / (count * nOut);
    }

    accuracy(x, labels, count) {
        const { out } = this.forwardBatch(x, count);
        const nOut = this.sizes[this.sizes.length - 1];
        let hits = 0;
        for (let b = 0; b < count; b++) {
            let best = 0;
            for (let o = 1; o < nOut; o++) if (out[b * nOut + o] > out[b * nOut + best]) best = o;
            if (best === labels[b]) hits++;
        }
        return hits / count;
    }

    // Install pretrained tensors named `${prefix}w0`, `${prefix}b0`, ...
    loadTensors(tensors, prefix = '') {
        for (let l = 0; l < this.depth; l++) {
            const w = tensors[`${prefix}w${l}`];
            const b = tensors[`${prefix}b${l}`];
            if (!w || !b) throw new Error(`missing tensor ${prefix}w${l}/${prefix}b${l}`);
            if (w.data.length !== this.weights[l].length) {
                throw new Error(`shape mismatch at layer ${l}: ${w.data.length} vs ${this.weights[l].length}`);
            }
            this.weights[l].set(w.data);
            this.biases[l].set(b.data);
        }
        return this;
    }
}
