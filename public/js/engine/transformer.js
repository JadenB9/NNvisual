import { affineForward, matmul, layernorm, gelu, softmaxRows } from './tensor.js';

// Decoder-only char transformer, inference with full capture. Pre-LN GPT
// block: x += attn(ln1(x)); x += mlp(ln2(x)). Tensor names (layer i):
//   tok_emb [V,d]  pos_emb [block,d]
//   l{i}_ln1_g/b   l{i}_wq/wk/wv/wo [d,d] + l{i}_bq/bk/bv/bo
//   l{i}_ln2_g/b   l{i}_fc1 [4d,d] + l{i}_fc1b   l{i}_fc2 [d,4d] + l{i}_fc2b
//   lnf_g/b        head [V,d]
// meta: { vocab, block, dModel, nHead, nLayer, chars }
export class Transformer {
    constructor(weights) {
        this.t = weights.tensors;
        const m = weights.meta;
        this.vocab = m.vocab;
        this.block = m.block;
        this.d = m.dModel;
        this.nHead = m.nHead;
        this.nLayer = m.nLayer;
        this.chars = m.chars; // index -> character
        this.stoi = {};
        for (let i = 0; i < this.chars.length; i++) this.stoi[this.chars[i]] = i;
    }

    encode(text) {
        const out = [];
        for (const ch of text) {
            if (ch in this.stoi) out.push(this.stoi[ch]);
        }
        return out;
    }

    decode(tokens) {
        return tokens.map((t) => this.chars[t]).join('');
    }

    // tokens: int array, length T <= block. Returns logits for every position
    // plus everything the page visualizes.
    forward(tokens) {
        const T = tokens.length;
        const d = this.d;
        const nHead = this.nHead;
        const dh = d / nHead;
        const t = this.t;

        const tokEmb = new Float32Array(T * d);
        const posEmb = new Float32Array(T * d);
        const x = new Float32Array(T * d);
        for (let i = 0; i < T; i++) {
            const te = tokens[i] * d;
            for (let c = 0; c < d; c++) {
                tokEmb[i * d + c] = t.tok_emb.data[te + c];
                posEmb[i * d + c] = t.pos_emb.data[i * d + c];
                x[i * d + c] = tokEmb[i * d + c] + posEmb[i * d + c];
            }
        }

        const layers = [];
        for (let li = 0; li < this.nLayer; li++) {
            const p = (n) => t[`l${li}_${n}`].data;
            const h = layernorm(x, p('ln1_g'), p('ln1_b'), T, d);
            const q = affineForward(h, p('wq'), p('bq'), T, d, d);
            const k = affineForward(h, p('wk'), p('bk'), T, d, d);
            const v = affineForward(h, p('wv'), p('bv'), T, d, d);

            const heads = [];
            const mixed = new Float32Array(T * d);
            for (let hd = 0; hd < nHead; hd++) {
                const off = hd * dh;
                // scaled causal scores
                const scores = new Float32Array(T * T).fill(-1e30);
                for (let i = 0; i < T; i++) {
                    for (let j = 0; j <= i; j++) {
                        let s = 0;
                        for (let c = 0; c < dh; c++) s += q[i * d + off + c] * k[j * d + off + c];
                        scores[i * T + j] = s / Math.sqrt(dh);
                    }
                }
                const probs = softmaxRows(scores, T, T);
                for (let i = 0; i < T; i++) {
                    for (let c = 0; c < dh; c++) {
                        let s = 0;
                        for (let j = 0; j <= i; j++) s += probs[i * T + j] * v[j * d + off + c];
                        mixed[i * d + off + c] = s;
                    }
                }
                heads.push({ scores, probs });
            }

            const attnOut = affineForward(mixed, p('wo'), p('bo'), T, d, d);
            for (let i = 0; i < T * d; i++) x[i] += attnOut[i];

            const h2 = layernorm(x, p('ln2_g'), p('ln2_b'), T, d);
            const m1 = affineForward(h2, p('fc1'), p('fc1b'), T, d, 4 * d);
            const g = gelu(m1);
            const m2 = affineForward(g, p('fc2'), p('fc2b'), T, 4 * d, d);
            for (let i = 0; i < T * d; i++) x[i] += m2[i];

            layers.push({ heads, mlpAct: g, attnOut, ln1: h });
        }

        const xf = layernorm(x, t.lnf_g.data, t.lnf_b.data, T, d);
        const logits = matmul(xf, transposeHead(t.head.data, this.vocab, d), T, d, this.vocab);
        return { tokens, tokEmb, posEmb, layers, final: xf, logits, T };
    }

    // Next-token distribution at the last position, with temperature/top-k.
    nextDistribution(tokens, temperature = 1, topK = 0) {
        const fwd = this.forward(tokens);
        const T = fwd.T;
        const V = this.vocab;
        const logits = fwd.logits.slice((T - 1) * V, T * V);
        const scaled = new Float32Array(V);
        const temp = Math.max(temperature, 1e-3);
        for (let i = 0; i < V; i++) scaled[i] = logits[i] / temp;
        if (topK > 0 && topK < V) {
            const order = [...scaled.keys()].sort((a, b) => scaled[b] - scaled[a]);
            for (let r = topK; r < V; r++) scaled[order[r]] = -1e30;
        }
        const probs = softmaxRows(scaled, 1, V);
        return { probs, logits, fwd };
    }

    sample(probs, rand) {
        let r = rand();
        for (let i = 0; i < probs.length; i++) {
            r -= probs[i];
            if (r <= 0) return i;
        }
        return probs.length - 1;
    }
}

// head is stored [V,d] row-per-vocab; matmul wants [d,V]
function transposeHead(head, V, d) {
    const out = new Float32Array(d * V);
    for (let v = 0; v < V; v++) {
        for (let c = 0; c < d; c++) out[c * V + v] = head[v * d + c];
    }
    return out;
}
