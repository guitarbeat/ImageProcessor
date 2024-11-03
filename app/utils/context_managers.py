"""Context managers for visualization."""
from contextlib import contextmanager
import matplotlib.pyplot as plt


@contextmanager
def figure_context(*args, **kwargs):
    """Context manager for creating and closing matplotlib figures."""
    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)


@contextmanager
def visualization_context():
    """Context manager for consistent visualization settings."""
    with figure_context() as fig:
        ax = fig.add_subplot(111)
        yield fig, ax
        plt.tight_layout()