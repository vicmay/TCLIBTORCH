#!/usr/bin/env tclsh

# Load the LibTorch TCL extension
if {[catch {load ./libtorchtcl.so} err]} {
    puts "Error loading libtorchtcl.so: $err"
    exit 1
}

puts "Testing LibTorch TCL Normalization Layers - Batch Implementation"
puts "=================================================================="

# Test counter
set test_count 0
set passed_count 0

proc test_command {name cmd expected_result} {
    global test_count passed_count
    incr test_count
    
    puts -nonewline "Testing $name... "
    if {[catch {eval $cmd} result]} {
        puts "FAILED: $result"
        return
    }
    
    if {$expected_result eq "success" || $result eq $expected_result} {
        puts "PASSED"
        incr passed_count
    } else {
        puts "FAILED: Expected '$expected_result', got '$result'"
    }
}

puts "\n=== Testing Local Response Normalization ==="
# Create a 3D tensor (batch, channels, spatial) for LRN
test_command "Local Response Norm" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set x [torch::tensor_reshape $x {1 2 4}]
    torch::local_response_norm $x 2 0.0001 0.75 1.0
} "success"

puts "\n=== Testing Cross-Map LRN2D ==="
# Create a 4D tensor (batch, channels, height, width) for 2D LRN
test_command "Cross-Map LRN2D" {
    set x2d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set x2d [torch::tensor_reshape $x2d {1 2 2 2}]
    torch::cross_map_lrn2d $x2d 2 0.0001 0.75 1.0
} "success"

puts "\n=== Testing Batch Norm 3D ==="
# Create a 5D tensor (batch, channels, depth, height, width) for 3D batch norm
test_command "Batch Norm 3D" {
    set x3d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set x3d [torch::tensor_reshape $x3d {1 1 2 2 2}]
    torch::batch_norm3d $x3d
} "success"

puts "\n=== Testing Instance Normalization ==="
# Test with appropriate tensor dimensions for each instance norm
test_command "Instance Norm 1D" {
    set x_inst1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set x_inst1d [torch::tensor_reshape $x_inst1d {1 2 3}]
    torch::instance_norm1d $x_inst1d
} "success"

test_command "Instance Norm 2D" {
    set x_inst2d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set x_inst2d [torch::tensor_reshape $x_inst2d {1 2 2 2}]
    torch::instance_norm2d $x_inst2d
} "success"

test_command "Instance Norm 3D" {
    set x_inst3d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set x_inst3d [torch::tensor_reshape $x_inst3d {1 1 2 2 2}]
    torch::instance_norm3d $x_inst3d
} "success"

puts "\n=== Testing RMS Normalization ==="
test_command "RMS Norm" {
    set x_rms [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set x_rms [torch::tensor_reshape $x_rms {2 3}]
    torch::rms_norm $x_rms {3}
} "success"

puts "\n=== Testing Spectral Normalization ==="
test_command "Spectral Norm" {
    set weight_matrix [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set weight_matrix [torch::tensor_reshape $weight_matrix {2 3}]
    torch::spectral_norm $weight_matrix
} "success"

puts "\n=== Testing Weight Normalization ==="
test_command "Weight Norm" {
    set weight_matrix2 [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set weight_matrix2 [torch::tensor_reshape $weight_matrix2 {2 3}]
    torch::weight_norm $weight_matrix2
} "success"

puts "\n=== Summary ==="
puts "Tests passed: $passed_count/$test_count"

if {$passed_count == $test_count} {
    puts "🎉 All normalization tests PASSED!"
    exit 0
} else {
    puts "❌ Some tests failed"
    exit 1
} 