// Flat Float32Array math. Shapes travel as explicit numbers; nothing here
// allocates inside a training loop if the caller passes `out` buffers.

// Y[b,o] = sum_i X[b,i] * W[o,i] + bias[o]   (W is row-per-output-unit)
export function affineForward(x, w, bias, batch, nIn, nOut, out) {
    out = out || new Float32Array(batch * nOut);
    for (let b = 0; b < batch; b++) {
        const xOff = b * nIn;
        const yOff = b * nOut;
        for (let o = 0; o < nOut; o++) {
            const wOff = o * nIn;
            let sum = bias ? bias[o] : 0;
            for (let i = 0; i < nIn; i++) sum += x[xOff + i] * w[wOff + i];
            out[yOff + o] = sum;
        }
    }
    return out;
}

// Given dY, accumulate dW/dB and (optionally) produce dX.
export function affineBackward(x, w, dy, batch, nIn, nOut, dw, db, dx) {
    for (let b = 0; b < batch; b++) {
        const xOff = b * nIn;
        const yOff = b * nOut;
        for (let o = 0; o < nOut; o++) {
            const g = dy[yOff + o];
            if (g === 0) continue;
            const wOff = o * nIn;
            db[o] += g;
            for (let i = 0; i < nIn; i++) dw[wOff + i] += g * x[xOff + i];
        }
    }
    if (dx) {
        dx.fill(0);
        for (let b = 0; b < batch; b++) {
            const xOff = b * nIn;
            const yOff = b * nOut;
            for (let o = 0; o < nOut; o++) {
                const g = dy[yOff + o];
                if (g === 0) continue;
                const wOff = o * nIn;
                for (let i = 0; i < nIn; i++) dx[xOff + i] += g * w[wOff + i];
            }
        }
    }
}

// A [m,k] * B [k,n] -> out [m,n] (used by attention; B in row-major)
export function matmul(a, b, m, k, n, out) {
    out = out || new Float32Array(m * n);
    out.fill(0);
    for (let i = 0; i < m; i++) {
        const aOff = i * k;
        const oOff = i * n;
        for (let p = 0; p < k; p++) {
            const av = a[aOff + p];
            if (av === 0) continue;
            const bOff = p * n;
            for (let j = 0; j < n; j++) out[oOff + j] += av * b[bOff + j];
        }
    }
    return out;
}

export function relu(x, out) {
    out = out || new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
    return out;
}

// dx = dy where the forward output was > 0
export function reluBackward(y, dy, dx) {
    for (let i = 0; i < y.length; i++) dx[i] = y[i] > 0 ? dy[i] : 0;
    return dx;
}

export function tanhAct(x, out) {
    out = out || new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = Math.tanh(x[i]);
    return out;
}

export function sigmoid(x, out) {
    out = out || new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = 1 / (1 + Math.exp(-x[i]));
    return out;
}

// tanh-approximation GELU, matching the numpy trainers.
const GELU_C = Math.sqrt(2 / Math.PI);
export function gelu(x, out) {
    out = out || new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
        const v = x[i];
        out[i] = 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)));
    }
    return out;
}

// Row-wise, numerically stable softmax over [rows, cols].
export function softmaxRows(x, rows, cols, out) {
    out = out || new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
        const off = r * cols;
        let max = -Infinity;
        for (let c = 0; c < cols; c++) if (x[off + c] > max) max = x[off + c];
        let sum = 0;
        for (let c = 0; c < cols; c++) {
            const e = Math.exp(x[off + c] - max);
            out[off + c] = e;
            sum += e;
        }
        for (let c = 0; c < cols; c++) out[off + c] /= sum;
    }
    return out;
}

// LayerNorm over the last dimension with scale/shift.
export function layernorm(x, gamma, beta, rows, cols, out) {
    out = out || new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
        const off = r * cols;
        let mean = 0;
        for (let c = 0; c < cols; c++) mean += x[off + c];
        mean /= cols;
        let variance = 0;
        for (let c = 0; c < cols; c++) {
            const d = x[off + c] - mean;
            variance += d * d;
        }
        variance /= cols;
        const inv = 1 / Math.sqrt(variance + 1e-5);
        for (let c = 0; c < cols; c++) {
            out[off + c] = (x[off + c] - mean) * inv * gamma[c] + beta[c];
        }
    }
    return out;
}

export function argmax(x, offset = 0, length = x.length - offset) {
    let best = offset;
    for (let i = offset + 1; i < offset + length; i++) if (x[i] > x[best]) best = i;
    return best - offset;
}
