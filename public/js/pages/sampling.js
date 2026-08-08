// Page-side shaping of the transformer's next-character distribution. The
// engine already applied temperature and top-k; these run after it, and every
// one of them reads the real probability array — nothing here invents a number.

// Nucleus (top-p): keep the smallest prefix of the descending-sorted
// distribution whose cumulative mass reaches p, renormalize that prefix, and
// leave every other character at exactly zero.
export function nucleus(probs, p) {
    if (!(p > 0) || p >= 1) return probs;
    const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
    const kept = new Float32Array(probs.length);
    let cum = 0;
    for (const i of order) {
        kept[i] = probs[i];
        cum += probs[i];
        if (cum >= p) break;
    }
    if (!(cum > 0)) return probs;
    for (let i = 0; i < kept.length; i++) kept[i] /= cum;
    return kept;
}

export function argmax(probs) {
    let best = 0;
    for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[best]) best = i;
    }
    return best;
}

// 1-based place of index i once the distribution is sorted descending.
export function rankOf(probs, i) {
    const p = probs[i];
    let rank = 1;
    for (let j = 0; j < probs.length; j++) {
        if (probs[j] > p) rank++;
    }
    return rank;
}

// how many characters still carry any mass — i.e. survived top-k and top-p.
export function keptCount(probs) {
    let n = 0;
    for (let i = 0; i < probs.length; i++) {
        if (probs[i] > 1e-9) n++;
    }
    return n;
}
