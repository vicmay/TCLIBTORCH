#!/usr/bin/env tclsh

# Load the LibTorch extension
load ./build/libtorchtcl.so

puts "Testing fractional pooling operations..."

# Test 2D fractional max pooling
puts "\n=== Testing torch::fractional_maxpool2d ==="

# Create a 4x4 input tensor for testing
set input_vals {}
for {set i 1} {$i <= 16} {incr i} {
    lappend input_vals [expr $i * 1.0]
}
set input [torch::tensor_create $input_vals float32 cpu 0]
set input [torch::tensor_reshape $input {1 1 4 4}]
puts "Input tensor (1x1x4x4):"
torch::tensor_print $input

set result [torch::fractional_maxpool2d $input {2 2}]
puts "Fractional maxpool2d result:"
torch::tensor_print $result

# Test with custom output ratio
set result_ratio [torch::fractional_maxpool2d $input {2 2} {0.75 0.75}]
puts "Fractional maxpool2d with ratio 0.75x0.75:"
torch::tensor_print $result_ratio

# Test 3D fractional max pooling
puts "\n=== Testing torch::fractional_maxpool3d ==="

# Create a 8x8x8 input tensor for testing (much larger to allow kernel size 2)
set input3d_vals {}
for {set i 1} {$i <= 512} {incr i} {
    lappend input3d_vals [expr $i * 1.0]
}
set input3d [torch::tensor_create $input3d_vals float32 cpu 0]
set input3d [torch::tensor_reshape $input3d {1 1 8 8 8}]
puts "Input 3D tensor (1x1x8x8x8):"
puts "Shape: [torch::tensor_shape $input3d]"

puts "Testing with kernel size {2 2 2} for 3D fractional pooling..."
set result3d [torch::fractional_maxpool3d $input3d {2 2 2}]
puts "Fractional maxpool3d result:"
puts "Result shape: [torch::tensor_shape $result3d]"
torch::tensor_print $result3d

# Test with custom output ratio
puts "Testing with ratio and kernel size {2 2 2}..."
set result3d_ratio [torch::fractional_maxpool3d $input3d {2 2 2} {0.6 0.6 0.6}]
puts "Fractional maxpool3d with ratio 0.6x0.6x0.6:"
puts "Result ratio shape: [torch::tensor_shape $result3d_ratio]"
torch::tensor_print $result3d_ratio

puts "\nFractional pooling tests completed successfully!" 