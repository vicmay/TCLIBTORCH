# torch::roi_pool / torch::roiPool

Performs Region-of-Interest (ROI) pooling on feature maps. This simplified implementation uses adaptive max-pooling to produce fixed-size feature maps for each ROI.

---

## Positional Syntax (legacy)
```tcl
set output [torch::roi_pool <input_tensor> <boxes_tensor> {h w} ?spatial_scale?]
```
* `input_tensor` – 4-D feature map tensor (NCHW).
* `boxes_tensor` – 2-D tensor of shape Nx5 where each row is `{batch_index x1 y1 x2 y2}`.
* `{h w}` – List specifying pooled output height and width.
* `spatial_scale` – Optional scaling factor to map box coordinates to feature-map scale (default `1.0`).

## Named Parameter Syntax (recommended)
```tcl
set output [torch::roiPool \
    -input       FEATS \
    -boxes       ROIS \
    -outputSize  {h w} \
    ?-spatialScale  S?]
```
Aliases: `-tensor` for `-input`, `-output_size` for `-outputSize`, `-spatial_scale` for `-spatialScale`.

## Parameters
| Name | Type | Description | Required |
|------|------|-------------|----------|
| `-input` | string | Feature-map tensor handle | ✓ |
| `-boxes` | string | ROI tensor handle | ✓ |
| `-outputSize` | list(2) | Output height & width | ✓ |
| `-spatialScale` | double | Coordinate scaling factor (default 1.0) | ✗ |

## Return Value
Returns a tensor handle containing pooled features for each ROI.

## Examples
```tcl
# Feature map (batch 1, 256 channels, 32×32)
set feats [torch::ones {1 256 32 32}]
# Single ROI covering center region
set boxes [torch::tensorCreate -data {0 8 8 24 24} -shape {1 5}]

# Positional usage
set pooled1 [torch::roi_pool $feats $boxes {7 7}]

# Named syntax
set pooled2 [torch::roiPool -input $feats -boxes $boxes -outputSize {7 7} -spatialScale 1.0]
```

## Compatibility
✅ Positional API retained • ✅ Named parameters added • ✅ camelCase alias registered 