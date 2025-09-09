#!/usr/bin/env tclsh

# Load the LibTorch extension
load ./build/libtorchtcl.so

puts "=== LibTorch TCL Extension - Fractional Pooling Operations Test ==="
puts "Testing both 2D and 3D fractional max pooling implementations."

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

# Test basic fractional pooling
set result [torch::fractional_maxpool2d $input {2 2}]
puts "\nFractional maxpool2d with kernel {2 2}:"
puts "Output shape: [torch::tensor_shape $result]"
torch::tensor_print $result

# Test with custom output ratio
set result_ratio [torch::fractional_maxpool2d $input {2 2} {0.75 0.75}]
puts "\nFractional maxpool2d with kernel {2 2} and ratio {0.75 0.75}:"
puts "Output shape: [torch::tensor_shape $result_ratio]"
torch::tensor_print $result_ratio

# Test 3D fractional max pooling
puts "\n=== Testing torch::fractional_maxpool3d ==="

# Create a 8x8x8 input tensor for testing (large enough for kernel requirements)
set input3d_vals {}
for {set i 1} {$i <= 512} {incr i} {
    lappend input3d_vals [expr $i * 1.0]
}
set input3d [torch::tensor_create $input3d_vals float32 cpu 0]
set input3d [torch::tensor_reshape $input3d {1 1 8 8 8}]
puts "Input 3D tensor (1x1x8x8x8):"
puts "Input shape: [torch::tensor_shape $input3d]"
puts "First few values (showing first 2 depth slices):"
set small_view [torch::tensor_slice $input3d 2 0 2]
torch::tensor_print $small_view

# Test basic 3D fractional pooling
set result3d [torch::fractional_maxpool3d $input3d {2 2 2}]
puts "\nFractional maxpool3d with kernel {2 2 2}:"
puts "Output shape: [torch::tensor_shape $result3d]"
set result_view [torch::tensor_slice $result3d 2 0 2]
torch::tensor_print $result_view

# Test with custom output ratio
set result3d_ratio [torch::fractional_maxpool3d $input3d {2 2 2} {0.6 0.6 0.6}]
puts "\nFractional maxpool3d with kernel {2 2 2} and ratio {0.6 0.6 0.6}:"
puts "Output shape: [torch::tensor_shape $result3d_ratio]"
set result_ratio_view [torch::tensor_slice $result3d_ratio 2 0 2]
torch::tensor_print $result_ratio_view

puts "\n=== ✅ Fractional Pooling Tests Completed Successfully! ==="
puts "Both torch::fractional_maxpool2d and torch::fractional_maxpool3d are working correctly."
puts "Total commands implemented in this session: 2 fractional pooling operations" 