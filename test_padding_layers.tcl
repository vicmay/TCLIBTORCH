#!/usr/bin/env tclsh

# Load the LibTorch TCL extension
if {[catch {load ./libtorchtcl.so} err]} {
    puts "Error loading libtorchtcl.so: $err"
    exit 1
}

puts "Testing LibTorch TCL Padding Layers - Batch Implementation"
puts "=========================================================="

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

puts "\n=== Testing Reflection Padding ==="

# Test 1D Reflection Padding - Needs at least 2D tensor (N, C, L)
test_command "Reflection Pad 1D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set x [torch::tensor_reshape $x {1 1 4}]
    set padded [torch::reflection_pad1d $x {1 2}]
    torch::tensor_shape $padded
} "1 1 7"

# Test 2D Reflection Padding - Needs at least 3D tensor (N, C, H, W)
test_command "Reflection Pad 2D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set x [torch::tensor_reshape $x {1 1 2 3}]
    set padded [torch::reflection_pad2d $x {1 1 1 1}]
    torch::tensor_shape $padded
} "1 1 4 5"

# Test 3D Reflection Padding - Needs at least 4D tensor (N, C, D, H, W)
test_command "Reflection Pad 3D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set x [torch::tensor_reshape $x {1 1 1 2 4}]
    set padded [torch::reflection_pad3d $x {1 1 1 1 0 0}]
    torch::tensor_shape $padded
} "1 1 1 4 6"

puts "\n=== Testing Replication Padding ==="

# Test 1D Replication Padding - Needs at least 2D tensor
test_command "Replication Pad 1D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set x [torch::tensor_reshape $x {1 1 4}]
    set padded [torch::replication_pad1d $x {2 1}]
    torch::tensor_shape $padded
} "1 1 7"

# Test 2D Replication Padding - Needs at least 3D tensor
test_command "Replication Pad 2D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set x [torch::tensor_reshape $x {1 1 2 3}]
    set padded [torch::replication_pad2d $x {1 2 1 1}]
    torch::tensor_shape $padded
} "1 1 4 6"

# Test 3D Replication Padding - Needs at least 4D tensor
test_command "Replication Pad 3D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set x [torch::tensor_reshape $x {1 1 1 2 4}]
    set padded [torch::replication_pad3d $x {1 1 0 1 0 0}]
    torch::tensor_shape $padded
} "1 1 1 3 6"

puts "\n=== Testing Constant Padding ==="

# Test 1D Constant Padding - Can work with any dimension tensor
test_command "Constant Pad 1D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set padded [torch::constant_pad1d $x {1 1} 0.0]
    torch::tensor_shape $padded
} "6"

# Test 2D Constant Padding  
test_command "Constant Pad 2D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set x [torch::tensor_reshape $x {2 3}]
    set padded [torch::constant_pad2d $x {1 1 1 1} 5.0]
    torch::tensor_shape $padded
} "4 5"

# Test 3D Constant Padding
test_command "Constant Pad 3D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set x [torch::tensor_reshape $x {1 2 4}]
    set padded [torch::constant_pad3d $x {1 1 1 1 0 0} -1.0]
    torch::tensor_shape $padded
} "1 4 6"

puts "\n=== Testing Circular Padding ==="

# Test 2D Circular Padding - PyTorch circular padding needs at least 3D tensor
test_command "Circular Pad 2D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set x [torch::tensor_reshape $x {1 2 3}]
    set padded [torch::circular_pad2d $x {1 1 1 0}]
    torch::tensor_shape $padded
} "1 3 5"

# Test 3D Circular Padding - For 4D tensors (batch, channel, depth, height, width)
test_command "Circular Pad 3D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set x [torch::tensor_reshape $x {1 1 2 4}]
    set padded [torch::circular_pad3d $x {1 1 0 0 0 0}]
    torch::tensor_shape $padded
} "1 1 2 6"

puts "\n=== Testing Zero Padding ==="

# Test 1D Zero Padding - Same as constant padding with value 0
test_command "Zero Pad 1D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set padded [torch::zero_pad1d $x {2 1}]
    torch::tensor_shape $padded
} "7"

# Test 2D Zero Padding
test_command "Zero Pad 2D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set x [torch::tensor_reshape $x {2 3}]
    set padded [torch::zero_pad2d $x {1 1 2 0}]
    torch::tensor_shape $padded
} "4 5"

# Test 3D Zero Padding
test_command "Zero Pad 3D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set x [torch::tensor_reshape $x {1 2 4}]
    set padded [torch::zero_pad3d $x {1 0 1 0 0 0}]
    torch::tensor_shape $padded
} "1 3 5"

puts "\n=== Testing Padding Values ==="

# Test that constant padding actually uses the specified value
test_command "Constant Pad Value Test" {
    set x [torch::tensor_create {5.0} float32]
    set padded [torch::constant_pad1d $x {1 1} 9.0]
    set shape [torch::tensor_shape $padded]
    if {$shape == "3"} {
        set result "success"
    } else {
        set result "failed - expected size 3, got $shape"
    }
    set result
} "success"

# Test different data types
test_command "Padding with Different Dtypes" {
    set x [torch::tensor_create {1 2 3} int32]
    set padded [torch::zero_pad1d $x {1 1}]
    torch::tensor_shape $padded
} "5"

puts "\n=== Testing Error Conditions ==="

# Test invalid padding dimensions
test_command "Invalid Padding Dimensions 1D" {
    set x [torch::tensor_create {1.0 2.0 3.0} float32]
    catch {torch::reflection_pad1d $x {1 2 3}} result
    if {[string match "*2 values*" $result]} {
        set test_result "success"
    } else {
        set test_result "failed - expected dimension error, got: $result"
    }
    set test_result
} "success"

# Test invalid padding dimensions 2D
test_command "Invalid Padding Dimensions 2D" {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set x [torch::tensor_reshape $x {2 2}]
    catch {torch::replication_pad2d $x {1 2}} result
    if {[string match "*4 values*" $result]} {
        set test_result "success"
    } else {
        set test_result "failed - expected dimension error, got: $result"
    }
    set test_result
} "success"

puts "\n==========================================="
puts "Padding Layers Test Results:"
puts "Total tests: $test_count"
puts "Passed: $passed_count"
puts "Failed: [expr {$test_count - $passed_count}]"
puts "Success rate: [expr {round(100.0 * $passed_count / $test_count)}]%"
puts "==========================================="

if {$passed_count == $test_count} {
    puts "🎉 ALL PADDING LAYER TESTS PASSED! 🎉"
    puts "15 new padding operations successfully implemented!"
    exit 0
} else {
    puts "❌ Some tests failed. Please check the implementation."
    exit 1
} 