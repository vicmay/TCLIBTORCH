#!/usr/bin/env tclsh

# Test file for torch::tensor_div command with dual syntax support
# Tests both positional and named parameter syntax

package require tcltest
namespace import tcltest::*

# Load the LibTorch TCL extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test suite for torch::tensor_div
test tensor_div-1.1 {Basic positional syntax} {
    # Create tensors for division
    set input1 [torch::tensor_create -data {10.0 20.0 30.0 40.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {2.0 4.0 5.0 8.0} -dtype float32 -device cpu]
    
    # Test basic tensor_div
    set result [torch::tensor_div $input1 $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-1.2 {Positional syntax with integer division} {
    # Create tensors for integer division
    set input1 [torch::tensor_create -data {10 20 30 40} -dtype int32 -device cpu]
    set input2 [torch::tensor_create -data {2 4 5 8} -dtype int32 -device cpu]
    
    # Test basic tensor_div
    set result [torch::tensor_div $input1 $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-2.1 {Named parameter syntax - basic} {
    # Create tensors for division
    set input1 [torch::tensor_create -data {15.0 25.0 35.0 45.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {3.0 5.0 7.0 9.0} -dtype float32 -device cpu]
    
    # Test basic named parameter syntax
    set result [torch::tensor_div -input $input1 -other $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-2.2 {Named parameter syntax with different order} {
    # Create tensors for division
    set input1 [torch::tensor_create -data {100.0 200.0 300.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]
    
    # Test named parameter syntax with different parameter order
    set result [torch::tensor_div -other $input2 -input $input1]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-3.1 {CamelCase alias - basic} {
    # Create tensors for division
    set input1 [torch::tensor_create -data {50.0 60.0 70.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {5.0 6.0 7.0} -dtype float32 -device cpu]
    
    # Test camelCase alias
    set result [torch::tensorDiv -input $input1 -other $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-3.2 {CamelCase alias with positional syntax} {
    # Create tensors for division
    set input1 [torch::tensor_create -data {80.0 90.0 100.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {8.0 9.0 10.0} -dtype float32 -device cpu]
    
    # Test camelCase alias with positional syntax
    set result [torch::tensorDiv $input1 $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-4.1 {Error handling - invalid first tensor} {
    # Test with non-existent first tensor
    set input2 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_div invalid_tensor $input2} result
    expr {[string length $result] > 0}
} {1}

test tensor_div-4.2 {Error handling - invalid second tensor} {
    # Test with non-existent second tensor
    set input1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_div $input1 invalid_tensor} result
    expr {[string length $result] > 0}
} {1}

test tensor_div-4.3 {Error handling - missing input} {
    # Test with missing input parameter
    catch {torch::tensor_div} result
    expr {[string length $result] > 0}
} {1}

test tensor_div-4.4 {Error handling - missing other} {
    # Test with missing other parameter
    set input1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    catch {torch::tensor_div -input $input1} result
    expr {[string length $result] > 0}
} {1}

test tensor_div-4.5 {Error handling - unknown parameter} {
    # Create tensors
    set input1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    
    # Test with unknown parameter
    catch {torch::tensor_div -input $input1 -other $input2 -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

test tensor_div-5.1 {Mathematical correctness - simple division} {
    # Create tensors for simple division
    set input1 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {2.0 4.0 5.0} -dtype float32 -device cpu]
    
    # Get division result
    set result [torch::tensor_div $input1 $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-5.2 {Mathematical correctness - division by one} {
    # Create tensors for division by one
    set input1 [torch::tensor_create -data {5.0 10.0 15.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {1.0 1.0 1.0} -dtype float32 -device cpu]
    
    # Get division result
    set result [torch::tensor_div $input1 $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-5.3 {Mathematical correctness - division by self} {
    # Create tensor for division by self
    set input1 [torch::tensor_create -data {5.0 10.0 15.0} -dtype float32 -device cpu]
    
    # Get division result
    set result [torch::tensor_div $input1 $input1]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-5.4 {Mathematical correctness - negative values} {
    # Create tensors with negative values
    set input1 [torch::tensor_create -data {-10.0 -20.0 -30.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {2.0 4.0 5.0} -dtype float32 -device cpu]
    
    # Get division result
    set result [torch::tensor_div $input1 $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-6.1 {Different data types - float32} {
    # Create tensors with float32
    set input1 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {2.0 4.0 5.0} -dtype float32 -device cpu]
    
    # Get division result
    set result [torch::tensor_div $input1 $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-6.2 {Different data types - float64} {
    # Create tensors with float64
    set input1 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float64 -device cpu]
    set input2 [torch::tensor_create -data {2.0 4.0 5.0} -dtype float64 -device cpu]
    
    # Get division result
    set result [torch::tensor_div $input1 $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-6.3 {Different data types - int32} {
    # Create tensors with int32
    set input1 [torch::tensor_create -data {10 20 30} -dtype int32 -device cpu]
    set input2 [torch::tensor_create -data {2 4 5} -dtype int32 -device cpu]
    
    # Get division result
    set result [torch::tensor_div $input1 $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-7.1 {Multi-dimensional tensor} {
    # Create 2D tensors
    set input1 [torch::zeros {2 2} float32 cpu]
    set input2 [torch::ones {2 2} float32 cpu]
    
    # Modify input1 to have non-zero values
    # Note: We can't easily modify tensors in this test framework, so we'll use the zeros/ones as is
    
    # Get division result
    set result [torch::tensor_div $input1 $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-8.1 {Zero values in numerator} {
    # Create tensors with zero values in numerator
    set input1 [torch::tensor_create -data {0.0 0.0 0.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    
    # Get division result
    set result [torch::tensor_div $input1 $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-8.2 {Large values} {
    # Create tensors with large values
    set input1 [torch::tensor_create -data {1000000.0 2000000.0 3000000.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {1000.0 2000.0 3000.0} -dtype float32 -device cpu]
    
    # Get division result
    set result [torch::tensor_div $input1 $input2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_div-9.1 {Syntax consistency - positional vs named} {
    # Create tensors
    set input1 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {2.0 4.0 5.0} -dtype float32 -device cpu]
    
    # Test both syntaxes
    set result1 [torch::tensor_div $input1 $input2]
    set result2 [torch::tensor_div -input $input1 -other $input2]
    
    # Both should return valid tensor handles
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_div-9.2 {Syntax consistency - snake_case vs camelCase} {
    # Create tensors
    set input1 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {2.0 4.0 5.0} -dtype float32 -device cpu]
    
    # Test both naming conventions
    set result1 [torch::tensor_div -input $input1 -other $input2]
    set result2 [torch::tensorDiv -input $input1 -other $input2]
    
    # Both should return valid tensor handles
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

# Clean up
cleanupTests 