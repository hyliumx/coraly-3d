from .coral import CoralDataset
from .modelnet40 import ModelNet40


def build_dataset(cfg, split):
    """Create a dataset from data.name in the config. To add a new dataset, just register it here."""
    d = dict(cfg["data"])
    name = d.pop("name")
    if name == "modelnet40":
        return ModelNet40(split=split, **d)
    if name == "coral":
        return CoralDataset(split=split, **d)
    raise ValueError(f"Unknown dataset: {name}")
