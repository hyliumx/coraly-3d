# Coraly-3D

Classifying coral growth forms directly from 3D point clouds with PointNet++.

## Repository layout

| Folder | Contents |
|---|---|
| [`pointcloud/`](pointcloud/) | PointNet++ point-cloud classification code (training, evaluation, configs). See [`pointcloud/README.md`](pointcloud/README.md) for setup and usage. |
| `mlc/` | Coral image dataset work (the Moorea Coral Reef LTER MLC data is large and is **not** included in the repo). |
| `papers/` | Reference papers (PointNet, PointNet++, coral annotation). |
| `reports/` | Project proposal and annotation reports. |
| `code/` | Reserved for additional code. |

## Quick start

```bash
cd pointcloud
# follow the setup steps in pointcloud/README.md
```

## Data

Datasets, model weights and logs are excluded via `.gitignore` because of their size
(`data/`, `mlc/dataset/`, `*.zip`, `*.npz`, `*.pth`, `logs/`). Download them separately.

## License

See [`pointcloud/LICENSE`](pointcloud/LICENSE). The model code is adapted from
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) (MIT).
