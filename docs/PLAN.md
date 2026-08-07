# NNvisual — rehaul plan

One page per network family, each the most honest interactive demonstration of that
architecture we can run in a browser. Nothing is faked: every activation, attention
weight, gradient, and denoising step shown on screen comes from a real forward or
backward pass executed in the page.

## Principles

1. **Genuine computation.** The visualization *is* the model running. Animations may
   pace what already happened (a pulse traveling edges), but never invent numbers.
2. **Zero runtime dependencies.** Vanilla ES modules + canvas. Compute uses
   `Float32Array` everywhere. Heavy work is sliced across frames to keep the UI live.
3. **Real data, real weights.** MNIST subsets ship with the site; reference models are
   trained offline by reproducible numpy scripts in `training/`, exported as fp16
   binaries, and verified in CI against golden fixtures the trainers emit.
4. **The user can always touch it.** Draw the input, edit the architecture, train on
   your own handwriting, drag the latent point, change the temperature.
5. **Keep the j4den theme.** Dark surface palette, Space Grotesk / Plex Sans / Plex
   Mono, muted greens/reds for signed values.

## Site map

```
index.html        chooser — "Visualize a neural network:" (4 cards, per mock)
mlp.html          Simple classification — Perceptron / MLP
cnn.html          Image detection — CNN
transformer.html  Language models — Transformer
diffusion.html    Image generation — Diffusion (+ latent-space / encoder tab)
```

Shared: `css/base.css` (tokens, chrome), `js/engine/` (math), `js/ui/` (draw box,
bars, heatmaps, pipeline scaffolding).

## Engine (`js/engine/`)

- `tensor.js` — Float32Array matmul / affine / elementwise; no allocation in hot loops.
- `f16.js` — fp16 ⇄ fp32 codec for weight files.
- `weights.js` — `.bin` loader: `NNW1` magic, JSON header (shapes, dtype), payload.
- `mlp.js` — dense nets: forward (capturing every activation), backward, SGD;
  the same class powers the MLP page, the autoencoder, and the diffusion denoiser.
- `conv.js` — conv2d + maxpool forward with per-map capture (backward only if
  in-browser fine-tuning ships; the CNN page is inference + arithmetic-level viz).
- `transformer.js` — inference with full capture: embeddings, per-head QK^T,
  softmax maps, head mixes, MLP activations, logits.
- `diffusion.js` — cosine/linear ᾱ schedule, forward noising `x_t = √ᾱ·x0 + √(1−ᾱ)·ε`,
  DDIM reverse sampler emitting every intermediate (x_t, ε̂, x̂0).
- `mnist.js` — packed dataset reader (`NND1` bins), batcher, and the canonical MNIST
  preprocessing for user drawings: bounding-box crop → scale longest side to 20 →
  paste into 28×28 at the center of mass. This is why drawn digits classify well.
- `store.js` — user dataset in localStorage (per-class example lists, import/export).

## Data + offline training (`training/`)

- `fetch_data.sh` — MNIST (cvdf mirror) + tiny-shakespeare into `training/data/`
  (gitignored).
- `pack_mnist.py` — 6,000 train / 1,000 test subset → `public/data/mnist-*.bin`
  (u8 pixels + labels), plus a 200-image test fixture for CI accuracy gates.
- `common.py` — numpy layers (affine, relu, layernorm, attention, adam) with a
  gradient-check harness; every trainer asserts its gradients numerically first.
- `train_mlp.py` — 784-16-16-10 reference classifier (≈94% test).
- `train_cnn.py` — conv(1→8,3×3)-pool-conv(8→16,3×3)-pool-dense10 (≈98% test).
- `train_transformer.py` — char-level tiny-shakespeare: block 64, d=64, 2 layers,
  4 heads (≈115k params). Exports vocab + weights + a golden logits fixture.
- `train_autoencoder.py` — 784-128-32-2-32-128-784, MSE; the 2-D bottleneck is the
  latent-space demo.
- `train_diffusion.py` — class-conditional ε-predictor MLP (x_t ⊕ t-embed ⊕ class
  embed → 512 → 512 → 784), cosine schedule, T=200 train / DDIM ~30 sample.
- Every trainer writes `public/data/weights/*.bin` + `tests/fixtures/*` goldens
  (same input → same output, asserted in JS to ~1e-3, proving the JS engine matches
  the trainer bit-for-purpose).

## The four demos

### 1. Simple classification — `mlp.html`

Layout: input rail | pipeline stage (the big scaling playground) | output rail;
tabs under the input rail: **Run · Train · Your data**.

- **Input**: 28×28 draw box (soft brush, MNIST-style grayscale), clear/undo, a strip
  of real test digits to load, and a live "what the network sees" preview showing the
  centered/normalized 28×28 after preprocessing.
- **Pipeline**: the 28×28 grid's lit pixels unroll — animated — into the 784-value
  input line (hover any cell: its exact value), which feeds the layer columns. Hidden
  neurons render as circles filled by live activation; edges show only the top-K
  |w·a| contributions (signed green/red, width ∝ magnitude) so the picture stays
  legible; a fast pulse sweeps input→output on each classify. Hover a hidden neuron:
  its incoming weight vector rendered as a 28×28 heatmap — "what this neuron looks
  for" — plus its bias and current pre/post-activation numbers.
- **Output**: 10 softmax bars with exact probabilities, argmax highlighted; toggle to
  raw logits.
- **Train tab**: architecture editor (1–3 hidden layers, 4–32 neurons each), LR /
  batch controls, train·pause·step on the real 6k MNIST subset in-page (typed-array
  SGD, several epochs/min), live loss + test-accuracy charts, weight heatmaps
  visibly organizing as training runs. Concept strips, driven by the real run:
  **gradient descent** (loss-bowl with the ball taking the actual step sizes; LR
  slider shows overshoot) and **backpropagation** (after a step, the error signal
  replays backward through the same edges with signed deltas).
- **Your data tab**: draw digits, label 0–9, see per-class counts; modes: MNIST /
  MNIST + yours / yours only; retrain live and watch accuracy on *your* held-out
  strokes; export/import JSON; persists in localStorage.

### 2. Image detection — `cnn.html`

Same draw box + sample strip feeding the pretrained convnet, every stage live:

- Stage lane: input 28×28 → **conv1** (8 learned 3×3 kernels drawn as heatmaps;
  8 feature maps, 26×26) → ReLU → **maxpool** (13×13, hover shows the 2×2 window
  that won) → **conv2** (16 maps) → pool (5×5) → **flatten** (400-strip) → dense →
  softmax bars.
- **Scan mode**: slow-motion convolution — the 3×3 window sweeps the input while the
  side panel shows the actual patch values × kernel values → sum, painting the
  feature map pixel by pixel. The single best way to teach convolution.
- Hover any feature-map pixel → its receptive field lights up on the input.
- **Kernel playground**: apply hand-editable kernels (edge / sharpen / blur / custom
  grid) to your drawing, separate from the trained net, to grok the operation itself.

### 3. Language models — `transformer.html`

A real 115k-param char transformer (trained on Shakespeare) generating live:

- Prompt box (seeded "ROMEO:"), temperature + top-k, **step** (one token, fully
  staged) and **autoplay**.
- Stage lanes: text → token chips with ids → **embedding** tiles (64-d as 8×8 heat
  tiles) + sinusoidal **positional** lane → per-layer, per-head **causal attention
  matrix** (the triangle, live) and an **arc diagram** over the token row showing
  where the next-token position attends; select any pair to see q·k, the scale, and
  its softmax share → MLP block activations → **logits → top-10 probability bars**
  → the sampled char animates onto the text.
- **Embeddings panel**: PCA of the 65 char embeddings to 2-D, hover a char to see
  its neighbors — spaces/newline cluster, digits cluster, vowels cluster.
- **Cosine similarity explorer**: pick two tokens (or drag two free vectors): the
  angle drawn, cos value live, tied to "attention scores are scaled dot products".

### 4. Image generation — `diffusion.html` (tabs: Diffusion · Latent space)

- **Forward process**: pick a real digit or draw one; drag t from 0→T and watch
  `x_t = √ᾱ·x0 + √(1−ᾱ)·ε` computed live, with the ᾱ schedule chart and the exact
  coefficients printed.
- **Reverse process**: choose a class 0–9, steps (10–50), seed; **Generate** runs
  real DDIM in-page, painting every step: current x_t, the model's predicted noise
  ε̂, and the implied x̂0, plus a film strip of intermediates you can scrub.
- **Latent space** (the encoder demo): the autoencoder's 2-D bottleneck. The 1k test
  digits are encoded live at load into a class-colored scatter; encode your drawing
  and watch its point land; **drag anywhere** in the plane and the decoder renders
  the morphing digit under your cursor; interpolate between two digits.

## Concept coverage

| Concept | Where |
|---|---|
| Gradient descent | MLP Train tab: loss bowl + real step sizes, LR overshoot |
| Backpropagation | MLP Train tab: signed error flow replayed on the real edges |
| Convolution | CNN scan mode + kernel playground |
| Attention / Transformers | Transformer page: QK^T maps, arcs, scaled dot products |
| Embeddings | Transformer embedding map + autoencoder latent plane |
| Cosine similarity | Transformer explorer panel |
| Encoders | Autoencoder tab (encode → 2-D → decode, draggable) |
| Diffusion | Forward + reverse process tabs with live schedule math |
| Softmax / loss | Output bars + live loss charts on every page |

## Delivery phases (one PR each, CI green, squash-merge)

1. Chooser page + this plan (old playground removed).
2. Engine core + full test suite (gradient checks, fp16 codec, preprocessing).
3. Datasets, training scripts, shipped weights, golden-fixture CI gates.
4. MLP page. 5. CNN page. 6. Transformer page. 7. Diffusion + latent page.
8. Polish (hover micro-previews, reduced motion, README), full browser walkthrough,
   sync to j4den.com, live verification.

## Performance budgets

- First paint of any page < 1s on broadband; dataset/weight bins lazy-load with a
  progress readout (MNIST subset ≈ 5.5 MB, brotli-compressed on the wire).
- MLP live training ≥ 3 epochs/min on the 6k subset (M1 baseline) at 60fps UI.
- Diffusion 30-step sample ≤ ~3s, painting per-step.
- All pages usable at 360px width; `prefers-reduced-motion` honored everywhere.

## Testing gates (CI-required)

- Engine unit tests + numerical gradient checks (dense; conv if backward ships).
- Golden fixtures: JS forward == numpy forward (~1e-3) for MLP, CNN, transformer,
  autoencoder, diffusion step.
- Shipped-weight quality: reference MLP ≥ 90% and CNN ≥ 95% on the 200-image
  fixture; transformer golden logits match; sampler reproduces the seeded fixture.
- Preprocessing tests (centering, scaling, value ranges) since drawn-digit accuracy
  depends on it.
