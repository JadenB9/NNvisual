"""Pack stratified MNIST subsets into the NND1 files the site ships.

  public/data/mnist-train.bin    6,000 train images (600 per class)
  public/data/mnist-test.bin     1,000 test images (100 per class)
  tests/fixtures/mnist-200.bin     200 further test images, disjoint from
                                   the shipped test set — the CI accuracy gate
"""

from pathlib import Path

import numpy as np

from common import read_idx_images, read_idx_labels, write_nnd1

root = Path(__file__).resolve().parent.parent
data = Path(__file__).resolve().parent / "data"


def stratified(images, labels, per_class, skip=0):
    """Take per_class of each digit, skipping the first `skip` of each."""
    picks = []
    for d in range(10):
        idx = np.flatnonzero(labels == d)[skip:skip + per_class]
        assert len(idx) == per_class, f"class {d}: only {len(idx)}"
        picks.append(idx)
    order = np.concatenate(picks)
    rng = np.random.default_rng(1234)
    rng.shuffle(order)
    return images[order], labels[order]


train_x = read_idx_images(data / "train-images-idx3-ubyte.gz")
train_y = read_idx_labels(data / "train-labels-idx1-ubyte.gz")
test_x = read_idx_images(data / "t10k-images-idx3-ubyte.gz")
test_y = read_idx_labels(data / "t10k-labels-idx1-ubyte.gz")

x, y = stratified(train_x, train_y, 600)
write_nnd1(root / "public/data/mnist-train.bin", x, y)

x, y = stratified(test_x, test_y, 100)
write_nnd1(root / "public/data/mnist-test.bin", x, y)

x, y = stratified(test_x, test_y, 20, skip=100)
write_nnd1(root / "tests/fixtures/mnist-200.bin", x, y)
