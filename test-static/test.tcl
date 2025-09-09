#!/usr/bin/tclsh

# Load the library
load ./libtorchtcl.so

# The library is now statically linked, no need to load it

puts "1. Basic Tensor Creation with Different Types"
puts "-------------------------------------------"
# Create tensors with different data types
set t1 [torch::tensor_create {1 2 3 4 5} "float32"]
puts "Float32 tensor: [torch::tensor_print $t1]"
puts "Type: [torch::tensor_dtype $t1]"

set t2 [torch::tensor_create {1 2 3 4 5} "float64"]
puts "\nFloat64 tensor: [torch::tensor_print $t2]"
puts "Type: [torch::tensor_dtype $t2]"

set t3 [torch::tensor_create {1 2 3 4 5} "int32"]
puts "\nInt32 tensor: [torch::tensor_print $t3]"
puts "Type: [torch::tensor_dtype $t3]"

puts "\n2. Autograd and Gradient Computation"
puts "-----------------------------------"
# Create tensor with gradient tracking
set x [torch::tensor_create {1 2 3} "float32" "cpu" 1]
puts "Requires gradient: [torch::tensor_requires_grad $x]"

# Create another tensor for computation
set y [torch::tensor_create {2 3 4} "float32" "cpu" 1]

# Perform operations
set z [torch::tensor_mul $x $y]
puts "\nz = x * y: [torch::tensor_print $z]"

# Compute sum for scalar output
set sum [torch::tensor_sum $z]
puts "sum(z): [torch::tensor_print $sum]"

# Backward pass
torch::tensor_backward $sum
puts "\nGradient of x: [torch::tensor_print [torch::tensor_grad $x]]"
puts "Gradient of y: [torch::tensor_print [torch::tensor_grad $y]]"

puts "\n3. Advanced Operations"
puts "---------------------"
# Create tensor for advanced ops
set data [torch::tensor_create {-2 -1 0 1 2}]

puts "Original: [torch::tensor_print $data]"
puts "Abs: [torch::tensor_print [torch::tensor_abs $data]]"
puts "Exp: [torch::tensor_print [torch::tensor_exp $data]]"
puts "Sigmoid: [torch::tensor_print [torch::tensor_sigmoid $data]]"
puts "ReLU: [torch::tensor_print [torch::tensor_relu $data]]"
puts "Tanh: [torch::tensor_print [torch::tensor_tanh $data]]"

puts "\n4. Reduction Operations"
puts "----------------------"
set nums [torch::tensor_create {1 2 3 4 5}]
puts "Tensor: [torch::tensor_print $nums]"
puts "Sum: [torch::tensor_print [torch::tensor_sum $nums]]"
puts "Mean: [torch::tensor_print [torch::tensor_mean $nums]]"
puts "Max: [torch::tensor_print [torch::tensor_max $nums]]"
puts "Min: [torch::tensor_print [torch::tensor_min $nums]]"

puts "\n5. CUDA Support (if available)"
puts "-----------------------------"
# Try moving tensor to CUDA
set cpu_tensor [torch::tensor_create {1 2 3 4 5}]
puts "Original device: [torch::tensor_device $cpu_tensor]"

set cuda_tensor [torch::tensor_to $cpu_tensor "cuda"]
puts "After moving to CUDA: [torch::tensor_device $cuda_tensor]"

puts "\n6. Neural Network Layer Tests"
puts "------------------------------"

# Create a linear layer
set linear [torch::linear 2 3 1]
puts "Created linear layer: #$linear"

# Create input tensor
set input [torch::tensor_create {1.0 2.0} "float32" "cpu" 1]
puts "Input tensor: [torch::tensor_print $input]"

# Forward pass through the layer
set output [torch::layer_forward $linear $input]
puts "Output tensor: [torch::tensor_print $output]"

# Compute gradients
set sum [torch::tensor_sum $output]
torch::tensor_backward $sum
puts "Input gradients: [torch::tensor_print [torch::tensor_grad $input]]"

puts "\n7. Convolutional Neural Network Tests"
puts "------------------------------------"

# Create a simple CNN
set conv1 [torch::conv2d 1 16 3 1 1 1]
puts "Created Conv2d layer: $conv1"

set pool1 [torch::maxpool2d 2 2]
puts "Created MaxPool2d layer: $pool1"

set dropout1 [torch::dropout 0.5]
puts "Created Dropout layer: $dropout1"

# Create a 1x1x28x28 input tensor (like MNIST)
set input_data {}
for {set i 0} {$i < 784} {incr i} {
    lappend input_data [expr {double(rand()) / double(0x7fffffff)}]
}
set input [torch::tensor_create $input_data "float32" "cpu" 1]
set input [torch::tensor_reshape $input {1 1 28 28}]
puts "Input shape: [torch::tensor_print $input]"

# Forward through the network
set conv_out [torch::layer_forward $conv1 $input]
puts "After Conv2d: [torch::tensor_print $conv_out]"

set pool_out [torch::layer_forward $pool1 $conv_out]
puts "After MaxPool2d: [torch::tensor_print $pool_out]"

set dropout_out [torch::layer_forward $dropout1 $pool_out]
puts "After Dropout: [torch::tensor_print $dropout_out]"

puts "\n8. Optimizer Tests"
puts "-----------------"

# Create parameters and optimizer
set w1 [torch::tensor_create {0.1 0.2 0.3} "float32" "cpu" 1]
set w2 [torch::tensor_create {0.4 0.5 0.6} "float32" "cpu" 1]
puts "Parameters before update:"
puts "w1: [torch::tensor_print $w1]"
puts "w2: [torch::tensor_print $w2]"

# Create SGD optimizer
set sgd [torch::optimizer_sgd [list $w1 $w2] 0.01 0.9 0.0001]
puts "Created SGD optimizer: $sgd"

# Create Adam optimizer
set adam [torch::optimizer_adam [list $w1 $w2] 0.001]
puts "Created Adam optimizer: $adam"

# Perform optimization step
torch::optimizer_zero_grad $sgd
set loss [torch::tensor_sum [torch::tensor_mul $w1 $w2]]
torch::tensor_backward $loss
torch::optimizer_step $sgd

puts "Parameters after SGD update:"
puts "w1: [torch::tensor_print $w1]"
puts "w2: [torch::tensor_print $w2]"

puts "\n9. Serialization Tests"
puts "---------------------"

# Save and load model state
torch::save_state $conv1 "conv1_state.pt"
puts "Saved Conv2d state to conv1_state.pt"

torch::load_state $conv1 "conv1_state.pt"
puts "Loaded Conv2d state from conv1_state.pt"

puts "\n10. Extended Tensor Operations Tests"
puts "--------------------------------"

# Test tensor reshape and permute
set t1 [torch::tensor_create {1 2 3 4 5 6} "float32"]
puts "Original tensor (1D):"
puts [torch::tensor_print $t1]

# First reshape to 2D
set t2 [torch::tensor_reshape $t1 {2 3}]
puts "\nReshaped tensor (2x3):"
puts [torch::tensor_print $t2]

# Now permute the 2D tensor
set t3 [torch::tensor_permute $t2 {1 0}]
puts "\nPermuted tensor (3x2):"
puts [torch::tensor_print $t3]

# Test tensor cat and stack
set t4 [torch::tensor_create {7 8 9 10 11 12} "float32"]
puts "\nSecond tensor (2x3):"
puts [torch::tensor_print $t4]

set t5 [torch::tensor_cat [list $t1 $t4] 0]
puts "\nConcatenated tensor (4x3):"
puts [torch::tensor_print $t5]

set t6 [torch::tensor_stack [list $t1 $t4] 0]
puts "\nStacked tensor (2x2x3):"
puts [torch::tensor_print $t6]

puts "\n11. Enhanced Neural Network Tests"
puts "------------------------------"

# Create a CNN with batch normalization and average pooling
set conv1 [torch::conv2d 1 16 3 1 1 1]
set bn1 [torch::batchnorm2d 16]
set pool1 [torch::avgpool2d 2]
set dropout1 [torch::dropout 0.5]

# Create a sequential model
set model [torch::sequential [list $conv1 $bn1 $pool1 $dropout1]]
puts "Created sequential model: $model"

# Create a test input (1x1x28x28)
set input_data {}
for {set i 0} {$i < 784} {incr i} {
    lappend input_data [expr {double(rand()) / double(0x7fffffff)}]
}
set input [torch::tensor_create $input_data "float32" "cpu" 1]
set input [torch::tensor_reshape $input {1 1 28 28}]
puts "\nInput shape: [torch::tensor_print $input]"

# Forward pass through the sequential model
set output [torch::layer_forward $model $input]
puts "\nOutput shape after sequential model:"
puts [torch::tensor_print $output]

puts "\n12. FFT and Complex Convolution Tests"
puts "-------------------------------------"

# Test FFT operations
puts "Testing FFT operations..."

# Create a 1D test tensor with proper shape for FFT
set input [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} {8}]
puts "Input tensor:"
puts [torch::tensor_print $input]

# Test 1D FFT along the signal dimension (dim 0)
set fft_result [torch::tensor_fft $input 0]
puts "1D FFT result:"
puts [torch::tensor_print $fft_result]

# Test inverse FFT
set ifft_result [torch::tensor_ifft $fft_result 0]
puts "1D IFFT result:"
puts [torch::tensor_print $ifft_result]

# Create a 2D test tensor
puts "\nCreating 2D tensor..."
set data_2d {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0}
puts "Data: $data_2d"

# First create as 1D tensor
set input_2d [torch::tensor_create $data_2d {16}]
puts "Input tensor (1D) shape: [torch::tensor_shape $input_2d]"
puts "Input tensor (1D):"
puts [torch::tensor_print $input_2d]

# Then create as 2D tensor
set input_2d_reshaped [torch::tensor_create $data_2d {4 4}]
puts "\nInput tensor (2D) shape: [torch::tensor_shape $input_2d_reshaped]"
puts "Input tensor (2D):"
puts [torch::tensor_print $input_2d_reshaped]

# Test 2D FFT on flattened tensor
puts "\nPerforming 2D FFT on flattened tensor..."
set fft2d_result [torch::tensor_fft2d $input_2d]
puts "FFT result shape: [torch::tensor_shape $fft2d_result]"
puts "FFT result:"
puts [torch::tensor_print $fft2d_result]

# Test 2D FFT on 2D tensor
puts "\nPerforming 2D FFT on 2D tensor..."
set fft2d_result_2d [torch::tensor_fft2d $input_2d_reshaped]
puts "FFT result shape: [torch::tensor_shape $fft2d_result_2d]"
puts "FFT result:"
puts [torch::tensor_print $fft2d_result_2d]

# Test 2D inverse FFT
puts "\nPerforming 2D inverse FFT..."
set ifft2d_result [torch::tensor_ifft2d $fft2d_result]
puts "IFFT result shape: [torch::tensor_shape $ifft2d_result]"
puts "IFFT result:"
puts [torch::tensor_print $ifft2d_result]

# Test complex convolution operations
puts "\nTesting complex convolution operations..."

# Test 2D convolution with proper shapes
puts "\nTesting Conv2d with weight setting"
puts "--------------------------------"

# Create input tensor
set input_conv2d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0} "float32"]
set input_conv2d [torch::tensor_reshape $input_conv2d {1 1 3 3}]
puts "Input tensor (3x3):"
puts [torch::tensor_print $input_conv2d]

# Create a 2D convolution layer with 1 input channel, 1 output channel, and 3x3 kernel
set conv2d_layer [torch::conv2d 1 1 3 1 1 1]  ;# in_channels, out_channels, kernel_size, stride, padding, bias

# Create weights for the convolution layer (Sobel operator for edge detection)
set kernel [torch::tensor_create {1.0 0.0 -1.0 2.0 0.0 -2.0 1.0 0.0 -1.0} "float32"]
set kernel [torch::tensor_reshape $kernel {1 1 3 3}]
puts "\nKernel tensor (Sobel operator):"
puts [torch::tensor_print $kernel]

# Set the weights of the convolution layer
torch::conv2d_set_weights $conv2d_layer $kernel

# Apply convolution
set output [torch::layer_forward $conv2d_layer $input_conv2d]
puts "\nConvolution result:"
puts [torch::tensor_print $output]

puts "\nComprehensive Conv2d Layer Tests"
puts "--------------------------------"

# Test 1: Basic 3x3 input with Sobel operator (horizontal edges)
puts "\nTest 1: 3x3 input with Sobel operator (horizontal edges)"
set input_3x3 [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0} "float32"]
set input_3x3 [torch::tensor_reshape $input_3x3 {1 1 3 3}]
puts "Input tensor (3x3):"
puts [torch::tensor_print $input_3x3]

set conv2d_3x3 [torch::conv2d 1 1 3 1 1 1]  ;# in_channels, out_channels, kernel_size, stride, padding, bias
set kernel_sobel [torch::tensor_create {1.0 0.0 -1.0 2.0 0.0 -2.0 1.0 0.0 -1.0} "float32"]
set kernel_sobel [torch::tensor_reshape $kernel_sobel {1 1 3 3}]
puts "\nSobel kernel:"
puts [torch::tensor_print $kernel_sobel]

torch::conv2d_set_weights $conv2d_3x3 $kernel_sobel
set output_3x3 [torch::layer_forward $conv2d_3x3 $input_3x3]
puts "\nOutput with Sobel operator:"
puts [torch::tensor_print $output_3x3]

# Test 2: 4x4 input with Gaussian blur kernel
puts "\nTest 2: 4x4 input with Gaussian blur kernel"
set input_4x4 [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} "float32"]
set input_4x4 [torch::tensor_reshape $input_4x4 {1 1 4 4}]
puts "Input tensor (4x4):"
puts [torch::tensor_print $input_4x4]

set conv2d_4x4 [torch::conv2d 1 1 3 1 1 1]
# Create normalized Gaussian kernel (1/16 * original values)
set kernel_gaussian [torch::tensor_create {0.0625 0.125 0.0625 0.125 0.25 0.125 0.0625 0.125 0.0625} "float32"]
set kernel_gaussian [torch::tensor_reshape $kernel_gaussian {1 1 3 3}]
puts "\nGaussian blur kernel:"
puts [torch::tensor_print $kernel_gaussian]

torch::conv2d_set_weights $conv2d_4x4 $kernel_gaussian
set output_4x4 [torch::layer_forward $conv2d_4x4 $input_4x4]
puts "\nOutput with Gaussian blur:"
puts [torch::tensor_print $output_4x4]

# Test 3: 5x5 input with different stride and padding
puts "\nTest 3: 5x5 input with stride=2, padding=0"
set input_5x5 [torch::tensor_create {
    1.0 2.0 3.0 4.0 5.0 
    6.0 7.0 8.0 9.0 10.0 
    11.0 12.0 13.0 14.0 15.0 
    16.0 17.0 18.0 19.0 20.0 
    21.0 22.0 23.0 24.0 25.0
} "float32"]
set input_5x5 [torch::tensor_reshape $input_5x5 {1 1 5 5}]
puts "Input tensor (5x5):"
puts [torch::tensor_print $input_5x5]

set conv2d_5x5 [torch::conv2d 1 1 3 2 0 1]  ;# stride=2, padding=0
set kernel_edge [torch::tensor_create {-1.0 -1.0 -1.0 -1.0 8.0 -1.0 -1.0 -1.0 -1.0} "float32"]
set kernel_edge [torch::tensor_reshape $kernel_edge {1 1 3 3}]
puts "\nEdge detection kernel:"
puts [torch::tensor_print $kernel_edge]

torch::conv2d_set_weights $conv2d_5x5 $kernel_edge
set output_5x5 [torch::layer_forward $conv2d_5x5 $input_5x5]
puts "\nOutput with edge detection (stride=2, padding=0):"
puts [torch::tensor_print $output_5x5]

puts "\n13. Basic Arithmetic Operations Tests"
puts "-----------------------------------"
# Test subtraction and matrix multiplication
set a [torch::tensor_create {1.0 2.0 3.0 4.0} "float32"]
set a [torch::tensor_reshape $a {2 2}]
puts "Tensor A (2x2):"
puts [torch::tensor_print $a]

set b [torch::tensor_create {5.0 6.0 7.0 8.0} "float32"]
set b [torch::tensor_reshape $b {2 2}]
puts "\nTensor B (2x2):"
puts [torch::tensor_print $b]

set sub_result [torch::tensor_sub $a $b]
puts "\nA - B:"
puts [torch::tensor_print $sub_result]

set matmul_result [torch::tensor_matmul $a $b]
puts "\nA @ B (matrix multiplication):"
puts [torch::tensor_print $matmul_result]

puts "\n14. Device Operations Tests"
puts "--------------------------"
# Create a tensor on CPU
set cpu_tensor [torch::tensor_create {1.0 2.0 3.0 4.0} "float32" "cpu"]
puts "Original tensor device: [torch::tensor_device $cpu_tensor]"

# Try moving to CUDA if available
if {[catch {
    set cuda_tensor [torch::tensor_to $cpu_tensor "cuda"]
    puts "After moving to CUDA: [torch::tensor_device $cuda_tensor]"
    
    # Move back to CPU
    set back_to_cpu [torch::tensor_to $cuda_tensor "cpu"]
    puts "After moving back to CPU: [torch::tensor_device $back_to_cpu]"
} err]} {
    puts "CUDA device not available: $err"
}
