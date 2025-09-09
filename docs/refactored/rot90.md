# torch::rot90

Rotates a tensor by 90 degrees in the plane specified by dims.

## Syntax

```tcl
# Positional syntax (backward compatible)
torch::rot90 input ?k? ?dims?

# Named parameter syntax
torch::rot90 -input tensor ?-k number? ?-dims list?

# camelCase alias
torch::rot90 -input tensor ?-k number? ?-dims list?
```

## Parameters

* `-input` (tensor, required)
  * The input tensor to rotate
  * Must be at least 2-dimensional

* `-k` (integer, optional)
  * Number of times to rotate by 90 degrees
  * Default: 1
  * Positive values rotate counterclockwise
  * Negative values rotate clockwise

* `-dims` (list of 2 integers, optional)
  * The plane to rotate in
  * Default: {0 1}
  * Must be a list of exactly 2 dimensions

## Return Value

Returns a new tensor rotated by 90 degrees k times in the specified plane.

## Examples

```tcl
# Create a 2x3 tensor
set tensor [torch::tensor_create {{1 2 3} {4 5 6}} float32]

# Basic rotation (90 degrees counterclockwise)
torch::rot90 $tensor
;# Returns: {{3 6} {2 5} {1 4}}

# Rotate 180 degrees (k=2)
torch::rot90 $tensor 2
;# Returns: {{6 5 4} {3 2 1}}

# Using named parameters
torch::rot90 -input $tensor -k 1 -dims {0 1}
;# Returns: {{3 6} {2 5} {1 4}}

# Rotate clockwise (negative k)
torch::rot90 -input $tensor -k -1
;# Returns: {{4 1} {5 2} {6 3}}
```

## Error Conditions

* Returns error if input tensor is invalid or not found
* Returns error if k is not a valid integer
* Returns error if dims is not a list of exactly 2 integers
* Returns error if any dimension in dims is out of range for the tensor

## See Also

* [torch::flip](flip.md)
* [torch::roll](roll.md)
* [torch::transpose](transpose.md) 