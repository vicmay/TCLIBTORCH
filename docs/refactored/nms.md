# torch::nms / torch::Nms

Performs non-maximum suppression (NMS) on a set of bounding boxes based on their scores and IoU (Intersection over Union) thresholds.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::nms -boxes tensor -scores tensor -iouThreshold value ?-scoreThreshold value?
torch::Nms -boxes tensor -scores tensor -iouThreshold value ?-scoreThreshold value?
```

### Positional Parameters (Legacy)
```tcl
torch::nms boxes scores iou_threshold ?score_threshold?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-boxes` | tensor | Required | Tensor of bounding boxes in format [x1, y1, x2, y2] |
| `-scores` | tensor | Required | Tensor of confidence scores for each box |
| `-iouThreshold` | float | Required | IoU threshold for filtering overlapping boxes (0.0 to 1.0) |
| `-scoreThreshold` | float | 0.0 | Minimum score threshold for considering a box |

## Description

The `torch::nms` command performs non-maximum suppression on a set of bounding boxes. It filters out boxes that have high overlap (measured by IoU) with other boxes that have higher confidence scores. This is commonly used in object detection to remove duplicate detections.

The boxes tensor should be a 2D tensor where each row represents a bounding box in [x1, y1, x2, y2] format, where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

The scores tensor should be a 1D tensor containing confidence scores corresponding to each box.

## Return Value

Returns a tensor containing the indices of the selected boxes after NMS.

## Examples

### Basic Usage with Named Parameters
```tcl
# Create boxes and scores
set boxes [torch::tensor_create -data {
    0.0 0.0 1.0 1.0
    0.1 0.1 1.1 1.1
    0.9 0.9 1.9 1.9
} -dtype float32]
set scores [torch::tensor_create -data {0.9 0.8 0.7} -dtype float32]

# Apply NMS with IoU threshold of 0.5
set indices [torch::nms -boxes $boxes -scores $scores -iouThreshold 0.5]
```

### Using Score Threshold
```tcl
# Apply NMS with both IoU and score thresholds
set indices [torch::nms -boxes $boxes -scores $scores -iouThreshold 0.5 -scoreThreshold 0.8]
```

### Legacy Positional Syntax
```tcl
# Same operation using positional syntax
set indices [torch::nms $boxes $scores 0.5 0.8]
```

## Error Handling

The command will raise an error if:
- Required parameters are missing
- Invalid tensor handles are provided
- IoU threshold is not between 0.0 and 1.0
- Boxes and scores tensors have incompatible shapes

## See Also

- `torch::box_iou` - Calculate IoU between two sets of boxes
- `torch::roi_align` - Region of Interest (RoI) align operation
- `torch::roi_pool` - Region of Interest (RoI) pooling operation
