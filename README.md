# LibTorch TCL Extension

A TCL extension that provides access to PyTorch's C++ library (LibTorch) functionality within TCL scripts.

## License

This extension uses LibTorch, which is licensed under the BSD 3-Clause License:

```
BSD 3-Clause License

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## Installation

### Prerequisites

- **TCL Development Libraries**: Install TCL development headers
  ```bash
  sudo apt install tcl-dev
  ```

- **CUDA Support**: For CUDA acceleration, install the NVIDIA CUDA toolkit
  ```bash
  sudo apt install nvidia-cuda-toolkit
  ```

- **LibTorch**: Ensure you have LibTorch installed in the `libtorch` directory
  ``` bash
  wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
  ```

### Ubuntu 25.04 Compatibility

This project is fully compatible with Ubuntu 25.04. The build system automatically uses the system-installed CUDA toolkit (`nvidia-cuda-toolkit` package) which resolves compatibility issues with newer system headers.

### Build Instructions

1. Clone or download the project
2. Ensure LibTorch is available in the `libtorch` directory
3. Build the extension:

```bash
mkdir build
cd build
cmake ..
make -j4
```

The build system will automatically:
- Detect and use the system CUDA installation (`/usr/bin/nvcc`)
- Configure compatibility flags for Ubuntu 25.04
- Link against the appropriate LibTorch libraries
- Generate `libtorchtcl.so` with full CUDA support

### Troubleshooting

If you encounter CUDA build issues:
1. Ensure `nvidia-cuda-toolkit` is installed via apt (not standalone CUDA)
2. Verify CUDA toolkit version: `nvcc --version`
3. The build system uses `/usr/bin/nvcc` and `/usr/lib/cuda` paths for compatibility

## Available Commands

All commands are in the `torch::` namespace.

### Tensor Creation and Display

#### torch::tensor_create
Creates a new tensor from a TCL list of numbers.

```tcl
set tensor [torch::tensor_create values ?shape? ?type? ?device? ?requires_grad?]
```

- **values**: A TCL list of numbers
- **shape** (optional): A list specifying tensor dimensions
- **type** (optional): Data type ("float32", "float64", "int32", "int64", "bool")
- **device** (optional): Device to store tensor ("cpu" or "cuda")
- **requires_grad** (optional): Enable gradient tracking (0 or 1)

#### torch::tensor_print
Prints the contents of a tensor.

```tcl
puts [torch::tensor_print $tensor]
```

### Arithmetic Operations

#### Basic Operations
- `torch::tensor_add`: Add two tensors
- `torch::tensor_sub`: Subtract two tensors
- `torch::tensor_mul`: Multiply two tensors element-wise
- `torch::tensor_div`: Divide two tensors element-wise
- `torch::tensor_matmul`: Matrix multiplication

```tcl
set result [torch::tensor_add $tensor1 $tensor2]
```

### Advanced Operations

#### Unary Operations
- `torch::tensor_abs`: Absolute value
- `torch::tensor_exp`: Exponential
- `torch::tensor_log`: Natural logarithm
- `torch::tensor_sqrt`: Square root
- `torch::tensor_sigmoid`: Sigmoid function
- `torch::tensor_relu`: ReLU activation
- `torch::tensor_tanh`: Hyperbolic tangent

```tcl
set result [torch::tensor_abs $tensor]
```

#### Reduction Operations
- `torch::tensor_sum`: Sum all elements
- `torch::tensor_mean`: Mean of all elements
- `torch::tensor_max`: Maximum value
- `torch::tensor_min`: Minimum value

```tcl
set sum [torch::tensor_sum $tensor]
```

### Autograd Support

#### Gradient Operations
- `torch::tensor_requires_grad`: Check if tensor requires gradients
- `torch::tensor_grad`: Get tensor's gradients
- `torch::tensor_backward`: Compute gradients

```tcl
set x [torch::tensor_create {1 2 3} {} "float32" "cpu" 1]
set y [torch::tensor_mul $x $x]
set sum [torch::tensor_sum $y]
torch::tensor_backward $sum
set grad [torch::tensor_grad $x]
```

### Device Management

#### Device Operations
- `torch::tensor_to`: Move tensor to specified device
- `torch::tensor_device`: Get tensor's current device

```tcl
set cuda_tensor [torch::tensor_to $tensor "cuda"]
```

### Property Queries

#### Tensor Properties
- `torch::tensor_dtype`: Get tensor's data type
- `torch::tensor_device`: Get tensor's device
- `torch::tensor_requires_grad`: Check gradient tracking
- `torch::tensor_grad`: Get gradient values

## Neural Network Layers

#### Layer Creation
- `torch::linear`: Create a linear (fully connected) layer
```tcl
set linear [torch::linear in_features out_features ?bias?]
```

- `torch::conv2d`: Create a 2D convolutional layer
```tcl
set conv [torch::conv2d in_channels out_channels kernel_size stride padding ?bias?]
```

- `torch::maxpool2d`: Create a 2D max pooling layer
```tcl
set pool [torch::maxpool2d kernel_size ?stride? ?padding?]
```

- `torch::avgpool2d`: Create a 2D average pooling layer
```tcl
set pool [torch::avgpool2d kernel_size ?stride? ?padding?]
```

- `torch::dropout`: Create a dropout layer
```tcl
set dropout [torch::dropout probability]
```

- `torch::batchnorm2d`: Create a 2D batch normalization layer
```tcl
set bn [torch::batchnorm2d num_features]
```

#### Sequential Models
Create a sequential model to chain layers together:
```tcl
set model [torch::sequential [list $conv1 $bn1 $pool1 $dropout1]]
```

#### Forward Pass
Run input through a layer or model:
```tcl
set output [torch::layer_forward $layer $input]
```

### Tensor Shape Operations

#### torch::tensor_reshape
Reshape a tensor to new dimensions:
```tcl
set reshaped [torch::tensor_reshape $tensor {batch_size channels height width}]
```

#### torch::tensor_permute
Permute tensor dimensions:
```tcl
set permuted [torch::tensor_permute $tensor {0 2 1}]
```

### Optimization

#### Optimizers
- `torch::optimizer_sgd`: Create SGD optimizer
```tcl
set optimizer [torch::optimizer_sgd [list $param1 $param2] learning_rate]
```

- `torch::optimizer_adam`: Create Adam optimizer
```tcl
set optimizer [torch::optimizer_adam [list $param1 $param2] learning_rate]
```

#### Optimizer Operations
- `torch::optimizer_step`: Update parameters
- `torch::optimizer_zero_grad`: Zero out gradients

### Complex Tensor Operations

### FFT Operations

The library provides Fast Fourier Transform (FFT) operations for both 1D and 2D tensors:

```tcl
# 1D FFT
set fft_result [torch::tensor_fft $tensor $dimension]

# 1D Inverse FFT
set ifft_result [torch::tensor_ifft $fft_tensor $dimension]

# 2D FFT
set fft2d_result [torch::tensor_fft2d $tensor]

# 2D Inverse FFT
set ifft2d_result [torch::tensor_ifft2d $fft_tensor]
```

Example usage:
```tcl
# Create a 1D signal
set signal [torch::tensor_create {1 2 3 4 5 6 7 8} {8}]

# Perform FFT
set freq_domain [torch::tensor_fft $signal 0]

# Transform back to time domain
set reconstructed [torch::tensor_ifft $freq_domain 0]
```

### Complex Convolution Operations

The library supports various convolution operations:

#### 1D Convolution
```tcl
# Basic 1D convolution
set result [torch::tensor_conv1d $input $kernel]

# 1D transpose convolution
set result [torch::tensor_conv_transpose1d $input $kernel]
```

Example usage:
```tcl
# Create input signal and kernel
set input [torch::tensor_create {1 2 3 4 5} {1 1 5}]
set kernel [torch::tensor_create {1 2 1} {1 1 3}]

# Apply convolution
set conv_result [torch::tensor_conv1d $input $kernel]
```

#### 2D Transpose Convolution
```tcl
# 2D transpose convolution
set result [torch::tensor_conv_transpose2d $input $kernel]
```

Example usage:
```tcl
# Create input and kernel tensors
set input [torch::tensor_create {
    {1 2 3}
    {4 5 6}
    {7 8 9}
} {1 1 3 3}]

set kernel [torch::tensor_create {
    {1 0}
    {0 1}
} {1 1 2 2}]

# Apply transpose convolution
set deconv_result [torch::tensor_conv_transpose2d $input $kernel]
```

Note: For all convolution operations, the input tensor must have the appropriate number of dimensions and shape for the operation type. The kernel tensor must also match the expected dimensions for the operation.

### Complete Neural Network Example

```tcl
# Create a CNN for MNIST
set conv1 [torch::conv2d 1 16 3 1 1]  # in_channels=1, out_channels=16, kernel=3, stride=1, padding=1
set bn1 [torch::batchnorm2d 16]
set pool1 [torch::maxpool2d 2]  # kernel=2, stride=2
set dropout1 [torch::dropout 0.5]

# Create sequential model
set model [torch::sequential [list $conv1 $bn1 $pool1 $dropout1]]

# Create input tensor (1x1x28x28)
set input_data {}
for {set i 0} {$i < 784} {incr i} {
    lappend input_data [expr {double(rand()) / double(0x7fffffff)}]
}
set input [torch::tensor_create $input_data "float32" "cpu" 1]
set input [torch::tensor_reshape $input {1 1 28 28}]

# Forward pass
set output [torch::layer_forward $model $input]
puts "Output shape: [torch::tensor_print $output]"

# Compute loss and backward
set loss [torch::tensor_sum $output]
torch::tensor_backward $loss

# Create optimizer and update parameters
set optimizer [torch::optimizer_sgd [list $conv1 $bn1] 0.01]
torch::optimizer_zero_grad $optimizer
torch::optimizer_step $optimizer
```

### Examples

#### Basic Tensor Operations
```tcl
# Create a 2x2 tensor
set a [torch::tensor_create {1 2 3 4} {2 2} "float32"]
set b [torch::tensor_create {5 6 7 8} {2 2} "float32"]

# Arithmetic operations
set sum [torch::tensor_add $a $b]
set diff [torch::tensor_sub $a $b]
set prod [torch::tensor_mul $a $b]  # Element-wise multiplication
set div [torch::tensor_div $a $b]
set matmul [torch::tensor_matmul $a $b]  # Matrix multiplication
```

#### Neural Network Example
```tcl
# Create a simple CNN
set conv1 [torch::conv2d 1 16 3 1 1 1]  # 3x3 conv, 1 input channel, 16 output channels
set bn1 [torch::batchnorm2d 16]
set pool1 [torch::maxpool2d 2]  # 2x2 max pooling
set dropout1 [torch::dropout 0.5]

# Create a sequential model
set model [torch::sequential [list $conv1 $bn1 $pool1 $dropout1]]

# Create input tensor (1x1x28x28 for MNIST-like data)
set input [torch::tensor_create $data {1 1 28 28} "float32"]

# Forward pass
set output [torch::layer_forward $model $input]
```

#### Optimization Example
```tcl
# Create parameters with gradients
set w1 [torch::tensor_create {0.1 0.2 0.3} "float32" "cpu" 1]
set w2 [torch::tensor_create {0.4 0.5 0.6} "float32" "cpu" 1]

# Create optimizer
set sgd [torch::optimizer_sgd [list $w1 $w2] 0.01]  # learning rate = 0.01
set adam [torch::optimizer_adam [list $w1 $w2] 0.001]  # learning rate = 0.001

# Training step
torch::optimizer_zero_grad $sgd
set loss [torch::tensor_sum [torch::tensor_mul $w1 $w2]]
torch::tensor_backward $loss
torch::optimizer_step $sgd
```

#### Image Processing Example
```tcl
# Create a 4x4 input image
set image [torch::tensor_create {
    1 2 3 4
    5 6 7 8
    9 10 11 12
    13 14 15 16
} {1 1 4 4} "float32"]

# Create Gaussian blur kernel
set kernel [torch::tensor_create {
    0.0625 0.125 0.0625
    0.125 0.25 0.125
    0.0625 0.125 0.0625
} {1 1 3 3} "float32"]

# Apply convolution
set conv2d_layer [torch::conv2d 1 1 3 1 1 1]
torch::conv2d_set_weights $conv2d_layer $kernel
set blurred [torch::layer_forward $conv2d_layer $image]
```

#### FFT Operations Example
```tcl
# Create a 1D signal
set signal [torch::tensor_create {1 2 3 4 5 6 7 8} {8} "float32"]

# Compute FFT
set fft_result [torch::tensor_fft $signal 0]

# Compute inverse FFT
set reconstructed [torch::tensor_ifft $fft_result 0]

# 2D FFT example
set image [torch::tensor_create {
    1 2 3 4
    5 6 7 8
    9 10 11 12
    13 14 15 16
} {4 4} "float32"]
set fft2d_result [torch::tensor_fft2d $image]
set ifft2d_result [torch::tensor_ifft2d $fft2d_result]
```

## Memory Management

Tensors are automatically managed by the extension using a global storage system with unique identifiers. Memory is properly cleaned up when:
- Tensors are no longer needed
- The TCL interpreter exits
- Errors occur during operations

## Error Handling

The extension provides comprehensive error handling for:
- Invalid tensor operations
- Shape mismatches
- Device availability (CUDA)
- Memory allocation failures
- Invalid parameter values

Error messages include detailed information about the cause and context of the error.

## Performance Considerations

1. **Memory Efficiency**
   - Tensors are stored in contiguous memory
   - Operations are performed in-place when possible
   - Automatic memory cleanup prevents memory leaks

2. **GPU Acceleration**
   - CUDA support for GPU acceleration when available
   - Automatic fallback to CPU when CUDA is not available
   - Efficient tensor movement between CPU and GPU

3. **Optimization**
   - Vectorized operations for CPU computations
   - Batch processing capabilities
   - Efficient memory reuse in sequential operations

## Testing

The extension includes a comprehensive test suite (`test.tcl`) that covers:
- Basic tensor operations
- Neural network layers
- Optimizers
- Device operations
- FFT and complex operations
- Error handling
- Memory management

Run the tests using:
```bash
tclsh test.tcl
```

## TODO

The following features are planned for future implementation:

### Loss Functions
- Cross Entropy Loss
- Mean Squared Error Loss
- Binary Cross Entropy Loss
- Custom loss function support

### Advanced Optimizers
- RMSprop
- Adagrad
- Adadelta
- AdamW

### Advanced Neural Network Layers
- LSTM/GRU layers for sequential data
- Embedding layers
- Transposed convolution layers (complete implementation and testing)
- Residual connections/blocks
- Multi-head attention layers

### Data Loading and Preprocessing
- Data loading utilities
- Data augmentation functions
- Batch processing utilities
- Dataset abstractions

### Model Utilities
- Model summary functionality
- Gradient clipping utilities
- Learning rate scheduling
- Early stopping functionality
- Enhanced model checkpointing

### Metrics and Monitoring
- Accuracy metrics
- Precision/recall metrics
- Confusion matrix utilities
- Training progress monitoring

### Advanced Features
- Distributed training support
- Quantization support
- Pruning utilities
- Mixed precision training

### Visualization
- Plotting utilities
- Tensor visualization tools
- Training progress visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
