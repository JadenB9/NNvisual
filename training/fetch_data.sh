#!/usr/bin/env bash
# Fetch the raw datasets into training/data/ (gitignored).
#   MNIST      — ossci mirror of the classic idx files
#   Shakespeare — karpathy's tiny_shakespeare corpus
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data

mnist=https://ossci-datasets.s3.amazonaws.com/mnist
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
    [ -f "data/$f" ] || curl -fsSL "$mnist/$f" -o "data/$f"
done

[ -f data/shakespeare.txt ] || curl -fsSL \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    -o data/shakespeare.txt

ls -la data
