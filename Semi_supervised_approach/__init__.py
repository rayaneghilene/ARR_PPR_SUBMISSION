__version__ = "0.2.0"


# Core data utilities
from Semi_supervised_approach.data import (
    embed_texts,
    build_knn_graph,
    build_bipartite_text_group,
    load_clean_dataset,
)

# Core models & trainers
from Semi_supervised_approach.models.training import (
    train_node_classification,
    train_link_prediction,
)

__all__ = [
    # Metadata
    "__version__",
    # "__author__",

    # Data utilities
    "embed_texts",
    "build_knn_graph",
    "build_bipartite_text_group",
    "load_clean_dataset",

    # Training utilities
    "train_node_classification",
    "train_link_prediction",
]