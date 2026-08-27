"""Single global-seed helper, called at the start of every entry point.

Keeping this in one place (rather than each script seeding ``random``/``numpy``
itself) is what makes the whole pipeline reproducible end-to-end: every
script and the test suite call ``set_seed`` the same way, and the seed value
is threaded through to every source of randomness (few-shot demo sampling,
component-order shuffles, train/val splits, Trainer seeding).

Only ``random``/``numpy`` are hard dependencies here so this module stays
importable in the GPU-less dev sandbox; torch/transformers seeding is a no-op
if those packages aren't installed.
"""

from __future__ import annotations

import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, full_determinism: bool = False) -> None:
    """Seed every source of randomness this project uses.

    Args:
        seed: the single seed value propagated everywhere.
        full_determinism: if True, also enables torch's stricter (and slower)
            deterministic algorithms. Bitwise reproducibility across different
            GPU models is not guaranteed even with this on; only across
            repeated runs on the *same* GPU model/driver/library versions.
            Document this caveat wherever results are reported.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if full_determinism:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:
        pass
