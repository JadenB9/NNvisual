import { relu, affineForward, softmaxRows } from './tensor.js';

// Convolution forward passes with full capture for the CNN page. Single
// image, stride 1, valid padding — exactly what the reference net needs,
// nothing generic enough to hide the arithmetic.

// input [C,H,W] flat; kernels [OC,C,KH,KW] flat; -> [OC,OH,OW]
export function conv2dForward(input, kernels, bias, C, H, W, OC, KH, KW, out) {
    const OH = H - KH + 1;
    const OW = W - KW + 1;
    out = out || new Float32Array(OC * OH * OW);
    for (let oc = 0; oc < OC; oc++) {
        const kBase = oc * C * KH * KW;
        const oBase = oc * OH * OW;
        for (let oy = 0; oy < OH; oy++) {
            for (let ox = 0; ox < OW; ox++) {
                let sum = bias ? bias[oc] : 0;
                for (let c = 0; c < C; c++) {
                    const iBase = c * H * W;
                    const kcBase = kBase + c * KH * KW;
                    for (let ky = 0; ky < KH; ky++) {
                        const iRow = iBase + (oy + ky) * W + ox;
                        const kRow = kcBase + ky * KW;
                        for (let kx = 0; kx < KW; kx++) {
                            sum += input[iRow + kx] * kernels[kRow + kx];
                        }
                    }
                }
                out[oBase + oy * OW + ox] = sum;
            }
        }
    }
    return out;
}

// 2x2 max pool, stride 2, floor semantics. Also returns, per output cell,
// the flat input index that won — the receptive-field hover uses it.
export function maxpool2x2(input, C, H, W) {
    const OH = Math.floor(H / 2);
    const OW = Math.floor(W / 2);
    const out = new Float32Array(C * OH * OW);
    const idx = new Int32Array(C * OH * OW);
    for (let c = 0; c < C; c++) {
        const iBase = c * H * W;
        const oBase = c * OH * OW;
        for (let oy = 0; oy < OH; oy++) {
            for (let ox = 0; ox < OW; ox++) {
                let best = -Infinity;
                let bestAt = -1;
                for (let dy = 0; dy < 2; dy++) {
                    for (let dx = 0; dx < 2; dx++) {
                        const at = iBase + (oy * 2 + dy) * W + ox * 2 + dx;
                        if (input[at] > best) { best = input[at]; bestAt = at; }
                    }
                }
                out[oBase + oy * OW + ox] = best;
                idx[oBase + oy * OW + ox] = bestAt;
            }
        }
    }
    return { out, idx };
}

// The reference MNIST convnet:
//   28x28 -> conv(1->8, 3x3) -> relu -> pool -> conv(8->16, 3x3) -> relu
//   -> pool -> flatten(400) -> dense(10) -> softmax
// Tensor names: c1w [8,1,3,3] c1b, c2w [16,8,3,3] c2b, fcw [10,400] fcb.
// Flatten order is (channel, row, col), matching numpy's C-order reshape.
export class ConvNet {
    loadTensors(tensors) {
        this.c1w = tensors.c1w.data;
        this.c1b = tensors.c1b.data;
        this.c2w = tensors.c2w.data;
        this.c2b = tensors.c2b.data;
        this.fcw = tensors.fcw.data;
        this.fcb = tensors.fcb.data;
        return this;
    }

    // x: Float32Array(784) in [0,1]. Returns every stage for the pipeline.
    forward(x) {
        const conv1 = conv2dForward(x, this.c1w, this.c1b, 1, 28, 28, 8, 3, 3); // [8,26,26]
        const relu1 = relu(conv1);
        const p1 = maxpool2x2(relu1, 8, 26, 26);                                // [8,13,13]
        const conv2 = conv2dForward(p1.out, this.c2w, this.c2b, 8, 13, 13, 16, 3, 3); // [16,11,11]
        const relu2 = relu(conv2);
        const p2 = maxpool2x2(relu2, 16, 11, 11);                               // [16,5,5]
        const flat = p2.out; // already C-order [16*5*5 = 400]
        const logits = affineForward(flat, this.fcw, this.fcb, 1, 400, 10);
        const probs = softmaxRows(logits, 1, 10);
        return { input: x, conv1, relu1, pool1: p1, conv2, relu2, pool2: p2, flat, logits, probs };
    }
}
