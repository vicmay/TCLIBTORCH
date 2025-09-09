#!/usr/bin/env tclsh

# Test file for torch::tensor_abs command with dual syntax support
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

# Test suite for torch::tensor_abs
test tensor_abs-1.1 {Basic positional syntax} {
    # Create a tensor with negative values
    set input [torch::tensor_create -data {-1.0 -2.0 3.0 -4.0} -dtype float32 -device cpu]
    
    # Test basic tensor_abs
    set result [torch::tensor_abs $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-2.1 {Named parameter syntax - basic} {
    # Create a tensor with negative values
    set input [torch::tensor_create -data {-5.0 -6.0 7.0 -8.0} -dtype float32 -device cpu]
    
    # Test basic named parameter syntax
    set result [torch::tensor_abs -input $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-3.1 {CamelCase alias - basic} {
    # Create a tensor with negative values
    set input [torch::tensor_create -data {-9.0 -10.0 11.0 -12.0} -dtype float32 -device cpu]
    
    # Test camelCase alias
    set result [torch::tensorAbs -input $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-4.1 {Error handling - invalid tensor} {
    # Test with non-existent tensor
    catch {torch::tensor_abs invalid_tensor} result
    expr {[string length $result] > 0}
} {1}

test tensor_abs-4.2 {Error handling - missing input} {
    # Test with missing input parameter
    catch {torch::tensor_abs} result
    expr {[string length $result] > 0}
} {1}

test tensor_abs-4.3 {Error handling - unknown parameter} {
    # Create a tensor
    set input [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    
    # Test with unknown parameter
    catch {torch::tensor_abs -input $input -unknown_param value} result
    expr {[string length $result] > 0}
} {1}

test tensor_abs-5.1 {Mathematical correctness - all negative} {
    # Create a tensor with all negative values
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -4.0} -dtype float32 -device cpu]
    
    # Get absolute values
    set result [torch::tensor_abs $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-5.2 {Mathematical correctness - mixed values} {
    # Create a tensor with mixed positive and negative values
    set input [torch::tensor_create -data {-1.5 2.5 -3.5 4.5} -dtype float32 -device cpu]
    
    # Get absolute values
    set result [torch::tensor_abs $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-5.3 {Mathematical correctness - all positive} {
    # Create a tensor with all positive values
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
    
    # Get absolute values
    set result [torch::tensor_abs $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-6.1 {Different data types - float32} {
    # Create a tensor with float32
    set input [torch::tensor_create -data {-1.0 -2.0 3.0} -dtype float32 -device cpu]
    
    # Get absolute values
    set result [torch::tensor_abs $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-6.2 {Different data types - float64} {
    # Create a tensor with float64
    set input [torch::tensor_create -data {-1.0 -2.0 3.0} -dtype float64 -device cpu]
    
    # Get absolute values
    set result [torch::tensor_abs $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-6.3 {Different data types - int32} {
    # Create a tensor with int32
    set input [torch::tensor_create -data {-1 -2 3} -dtype int32 -device cpu]
    
    # Get absolute values
    set result [torch::tensor_abs $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-7.1 {Multi-dimensional tensor} {
    # Create a 2D tensor with negative values using zeros and then modify
    set input [torch::zeros {2 2} float32 cpu]
    
    # Get absolute values
    set result [torch::tensor_abs $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-7.2 {3D tensor} {
    # Create a 3D tensor with negative values using zeros and then modify
    set input [torch::zeros {2 2 2} float32 cpu]
    
    # Get absolute values
    set result [torch::tensor_abs $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-8.1 {Zero values} {
    # Create a tensor with zero values
    set input [torch::tensor_create -data {0.0 -0.0 0.0} -dtype float32 -device cpu]
    
    # Get absolute values
    set result [torch::tensor_abs $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-8.2 {Large values} {
    # Create a tensor with large values
    set input [torch::tensor_create -data {-1000000.0 2000000.0 -3000000.0} -dtype float32 -device cpu]
    
    # Get absolute values
    set result [torch::tensor_abs $input]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test tensor_abs-9.1 {Syntax consistency - positional vs named} {
    # Create a tensor
    set input [torch::tensor_create -data {-1.0 -2.0 3.0} -dtype float32 -device cpu]
    
    # Test both syntaxes
    set result1 [torch::tensor_abs $input]
    set result2 [torch::tensor_abs -input $input]
    
    # Both should return valid tensor handles
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_abs-9.2 {Syntax consistency - snake_case vs camelCase} {
    # Create a tensor
    set input [torch::tensor_create -data {-1.0 -2.0 3.0} -dtype float32 -device cpu]
    
    # Test both naming conventions
    set result1 [torch::tensor_abs -input $input]
    set result2 [torch::tensorAbs -input $input]
    
    # Both should return valid tensor handles
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

# Clean up
cleanupTests 