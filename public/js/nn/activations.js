// Hidden-layer activations. `df` takes the *output* of the activation
// rather than its input, because that is what backprop already has in hand
// when it walks back through a layer.
export const activations = {
    tanh: {
        f: Math.tanh,
        df: (y) => 1 - y * y,
    },
    sigmoid: {
        f: (x) => 1 / (1 + Math.exp(-x)),
        df: (y) => y * (1 - y),
    },
    relu: {
        f: (x) => (x > 0 ? x : 0),
        df: (y) => (y > 0 ? 1 : 0),
    },
};
