"""
strategies package
------------------
EXPLICIT strategy manifest. Each active strategy is imported here exactly once;
the import triggers its @register decorator. This is the single place that
controls which strategies the application exposes.

To add a strategy:   write strategies/my_strategy.py, then add one line below.
To disable one:      comment out or remove its import line.
"""

from . import weinstein_setup   # noqa: F401
from . import sma_cross         # noqa: F401
from . import momentum_leaders  # noqa: F401

# Add new strategies here, e.g.:
# from . import my_strategy    # noqa: F401
