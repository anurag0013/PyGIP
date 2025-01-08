from .watermark import (
    graph_to_dataset,  # Add this utility class
    WatermarkGraph,
    GraphSAGE, 
    Defense,
    Watermark_sage
)

__all__ = [
    'graph_to_dataset',
    'WatermarkGraph', 
    'GraphSAGE',
    'Defense',
    'Watermark_sage'
]