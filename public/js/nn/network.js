import { activations } from './activations.js';

// Plain multilayer perceptron for 2-D binary classification. Weights are
// nested arrays (weights[l][j][i] = weight from unit i in layer l to unit j
// in layer l + 1) — no tensor library, every step is a readable loop.
//
// Hidden layers use the chosen activation; the output unit is always a
// sigmoid trained with binary cross-entropy, so the delta at the output
// reduces to (prediction - label).
export class Network {
    constructor(sizes, activation = 'tanh', rand = Math.random) {
        this.sizes = sizes.slice();
        this.activation = activation;
        this.weights = [];
        this.biases = [];
        for (let l = 0; l < sizes.length - 1; l++) {
            const scale = Math.sqrt(1 / sizes[l]); // Xavier-style fan-in scaling
            const w = [];
            const b = [];
            for (let j = 0; j < sizes[l + 1]; j++) {
                const row = [];
                for (let i = 0; i < sizes[l]; i++) row.push((rand() * 2 - 1) * scale);
                w.push(row);
                b.push(0);
            }
            this.weights.push(w);
            this.biases.push(b);
        }
    }

    // Returns the activation of every layer: acts[0] is the input and
    // acts[acts.length - 1] is [output], a probability in (0, 1).
    forward(input) {
        const hidden = activations[this.activation];
        const acts = [input.slice()];
        const last = this.weights.length - 1;
        for (let l = 0; l < this.weights.length; l++) {
            const prev = acts[l];
            const out = [];
            for (let j = 0; j < this.weights[l].length; j++) {
                const row = this.weights[l][j];
                let z = this.biases[l][j];
                for (let i = 0; i < row.length; i++) z += row[i] * prev[i];
                out.push(l === last ? 1 / (1 + Math.exp(-z)) : hidden.f(z));
            }
            acts.push(out);
        }
        return acts;
    }

    predict(x, y) {
        const acts = this.forward([x, y]);
        return acts[acts.length - 1][0];
    }

    // One SGD step over a batch; returns the mean loss before the update.
    trainBatch(batch, lr) {
        const hidden = activations[this.activation];
        const gw = this.weights.map((w) => w.map((row) => row.map(() => 0)));
        const gb = this.biases.map((b) => b.map(() => 0));
        let loss = 0;

        for (const p of batch) {
            const acts = this.forward([p.x, p.y]);
            const out = acts[acts.length - 1][0];
            loss += crossEntropy(out, p.label);

            let delta = [out - p.label];
            for (let l = this.weights.length - 1; l >= 0; l--) {
                const prev = acts[l];
                for (let j = 0; j < delta.length; j++) {
                    gb[l][j] += delta[j];
                    for (let i = 0; i < prev.length; i++) gw[l][j][i] += delta[j] * prev[i];
                }
                if (l === 0) break;
                const next = [];
                for (let i = 0; i < prev.length; i++) {
                    let sum = 0;
                    for (let j = 0; j < delta.length; j++) sum += this.weights[l][j][i] * delta[j];
                    next.push(sum * hidden.df(prev[i]));
                }
                delta = next;
            }
        }

        const scale = lr / batch.length;
        for (let l = 0; l < this.weights.length; l++) {
            for (let j = 0; j < this.weights[l].length; j++) {
                this.biases[l][j] -= scale * gb[l][j];
                for (let i = 0; i < this.weights[l][j].length; i++) {
                    this.weights[l][j][i] -= scale * gw[l][j][i];
                }
            }
        }
        return loss / batch.length;
    }

    loss(points) {
        let sum = 0;
        for (const p of points) sum += crossEntropy(this.predict(p.x, p.y), p.label);
        return sum / points.length;
    }

    accuracy(points) {
        let hits = 0;
        for (const p of points) {
            if ((this.predict(p.x, p.y) >= 0.5 ? 1 : 0) === p.label) hits++;
        }
        return hits / points.length;
    }
}

// Clamped so a saturated sigmoid cannot produce log(0) = -Infinity.
export function crossEntropy(p, label) {
    const eps = 1e-12;
    const q = Math.min(1 - eps, Math.max(eps, p));
    return -(label * Math.log(q) + (1 - label) * Math.log(1 - q));
}
