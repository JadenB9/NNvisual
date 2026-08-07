# NNvisual

Interactive neural network visualizer. Build a small multilayer perceptron in the
browser, train it on toy 2-D datasets, and watch it learn in real time — the
decision boundary, every weight in the network, and the loss curve all update live.

**Live:** [j4den.com/nnvisual](https://j4den.com/nnvisual/)

![NNvisual screenshot](docs/screenshot.jpg)

## What it does

- **Four datasets** — circles, moons, xor, and the classic two-arm spiral, each
  with adjustable label noise.
- **Editable architecture** — add and remove hidden layers, resize each one, and
  switch the activation (tanh / relu / sigmoid) while you experiment.
- **Live training** — minibatch SGD with hand-written backpropagation runs in the
  page. Play, pause, single-step, or reset with a fresh weight init.
- **Three synced views** — the network diagram colors every edge by weight sign
  and magnitude, the decision boundary shows what the model currently believes,
  and the loss chart shows how it got there.

## No dependencies

The entire engine — PRNG, datasets, forward pass, backprop, training loop — is
plain ES modules with zero runtime dependencies. No tensor library, no build
step, no bundler. The only dev dependency is ESLint.

That is a deliberate choice: every piece of math in this repo is meant to be
readable. The forward pass and backprop in
[`public/js/nn/network.js`](public/js/nn/network.js) are ordinary nested loops.

## Run it locally

```sh
npm run serve        # static server on http://localhost:4173
```

Any static file server works — the app is plain HTML/CSS/JS modules.

## Tests

```sh
npm install          # dev tooling only (eslint)
npm test             # node --test, no test framework
npm run lint
```

The suite covers the deterministic PRNG, activation derivatives against
numerical differentiation, dataset shape/balance/reproducibility, and the
network itself — including a gradient check that verifies backprop against
central-difference numerical gradients, and an end-to-end test that trains
xor to ≥ 90% accuracy.

## Project layout

```
public/            the deployable app (static, self-contained)
  js/nn/           engine: rng, activations, datasets, network
  js/app/          ui: controls, diagram, boundary, loss chart
  css/  fonts/     styling, self-hosted type
tests/             node:test suite for the engine
scripts/           sync-to-site.sh — copy public/ into the j4den.com repo
.github/           ci, codeql, dependabot, templates
```

## How it deploys

`public/` is copied into the [j4den.com](https://j4den.com) site repo at
`frontend/public/nnvisual/` (see `scripts/sync-to-site.sh`), where Cloudflare
Pages serves it at [j4den.com/nnvisual](https://j4den.com/nnvisual/). The page
ships with a strict Content-Security-Policy meta tag and works standalone too.

## Contributing

Work lands through pull requests — see [CONTRIBUTING.md](CONTRIBUTING.md).
Security notes live in [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
