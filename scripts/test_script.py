import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()


seed = args.seed
print("Hello Python world!")
print(f"Using seed: {seed}")
rng = np.random.default_rng(seed=args.seed)

x = rng.integers(1)
y = rng.integers(123)

print(f"Random numbers: {x}, {y}")
print("Done with Python!")
