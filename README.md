# NFS Oil Price Prediction

This repository contains a single Jupyter notebook, `GDELT_Model_Training.ipynb`,
which packages the full workflow for training a temporal graph neural network
(GNN) on 5–10 years of aligned graph snapshots with observed oil price changes.
The notebook houses the dataset loader, model definition, training utilities,
and a convenience function for running end-to-end experiments.

## Getting started

1. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Open `GDELT_Model_Training.ipynb` in Jupyter (or launch it in Google Colab via
the badge at the top of the notebook) and run the cells in order.

The notebook exposes a `run_training` helper that loads snapshot metadata,
performs chronological train/validation/test splits, trains the GraphSAGE model
with early stopping, and evaluates the held-out test set. Pass the path to your
metadata CSV—containing `timestamp`, `graph_path`, and `price_change` columns—
to begin training. Set `output_dir` to persist `training_history.json` and
`test_metrics.json` artifacts.

## Preparing data

Each row of the metadata CSV should describe a graph snapshot saved via
`torch.save` and the corresponding oil price change label:

```csv
timestamp,graph_path,price_change
2015-01-01,graphs/2015-01-01.pt,-0.8
2015-02-01,graphs/2015-02-01.pt,0.3
...
```

Graphs must provide node features (`data.x`) and edge indices (`data.edge_index`).
Optional attributes (edge features, node metadata, etc.) can also be stored in
the `torch_geometric.data.Data` object and are forwarded unchanged to the model.

To satisfy the temporal constraint enforced by `GraphSnapshotDataset`, ensure
the selected snapshots cover at least five consecutive years and no more than
ten years. The notebook’s helper arguments (`start_year`, `end_year`, and
`shuffle_within_year`) allow further control over the training window.

## Reproducibility tips

* Keep graph schemas consistent across snapshots so the model can reuse learned
  representations.
* Align oil price labels to a consistent horizon (e.g. month-over-month) for all
  snapshots before training.
* Record the training configuration and commit hash alongside exported metrics
  for future reference.
