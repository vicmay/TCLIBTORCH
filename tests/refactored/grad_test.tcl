#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Test 1: Basic positional syntax (backward compatibility)
test grad-1.1 {Basic positional syntax} {
    ;# Create test tensors
    set outputs [torch::tensor_create {2.0 3.0 4.0} float32 cpu true]
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    
    ;# Test positional syntax
    set result [torch::grad $outputs $inputs]
    
    ;# Check that result is a valid tensor handle
    expr {$result ne "" && [string match "tensor*" $result]}
} {1}

;# Test 2: Named parameter syntax
test grad-2.1 {Named parameter syntax} {
    ;# Create test tensors
    set outputs [torch::tensor_create {2.0 3.0 4.0} float32 cpu true]
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    
    ;# Test named parameter syntax
    set result [torch::grad -outputs $outputs -inputs $inputs]
    
    ;# Check that result is a valid tensor handle
    expr {$result ne "" && [string match "tensor*" $result]}
} {1}

;# Test 3: Named parameter syntax with alternative parameter names
test grad-2.2 {Named parameter syntax with alternative names} {
    ;# Create test tensors
    set outputs [torch::tensor_create {2.0 3.0 4.0} float32 cpu true]
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    
    ;# Test named parameter syntax with alternative names
    set result [torch::grad -output $outputs -input $inputs]
    
    ;# Check that result is a valid tensor handle
    expr {$result ne "" && [string match "tensor*" $result]}
} {1}

;# Test 4: Error handling - missing parameters (positional)
test grad-4.1 {Error handling - missing parameters (positional)} {
    ;# Test missing parameter should fail
    set result [catch {torch::grad} msg]
    expr {$result == 1}
} {1}

;# Test 5: Error handling - missing parameters (named)
test grad-4.2 {Error handling - missing parameters (named)} {
    ;# Test missing input parameter should fail
    set outputs [torch::tensor_create {2.0 3.0 4.0} float32 cpu true]
    set result [catch {torch::grad -outputs $outputs} msg]
    expr {$result == 1}
} {1}

;# Test 6: Error handling - unknown parameter
test grad-4.3 {Error handling - unknown parameter} {
    ;# Test unknown parameter should fail
    set outputs [torch::tensor_create {2.0 3.0 4.0} float32 cpu true]
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [catch {torch::grad -outputs $outputs -inputs $inputs -unknown value} msg]
    expr {$result == 1}
} {1}

;# Test 7: Consistency - both syntaxes should produce same result
test grad-7.1 {Consistency between syntaxes} {
    ;# Create test tensors
    set outputs [torch::tensor_create {2.0 3.0 4.0} float32 cpu true]
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    
    ;# Test both syntaxes
    set result1 [torch::grad $outputs $inputs]
    set result2 [torch::grad -outputs $outputs -inputs $inputs]
    
    ;# Both should return valid tensor handles
    expr {$result1 ne "" && $result2 ne "" && [string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

;# Test 8: Gradient properties
test grad-8.1 {Gradient result properties} {
    ;# Create test tensors
    set outputs [torch::tensor_create {2.0 3.0 4.0} float32 cpu true]
    set inputs [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    
    ;# Test gradient computation
    set result [torch::grad -outputs $outputs -inputs $inputs]
    
    ;# Check that result has requires_grad set
    set requires_grad [torch::tensor_requires_grad $result]
    expr {$requires_grad == 1}
} {1}

;# Test 9: Different data types
test grad-9.1 {Different data types} {
    ;# Create test tensors with different dtypes
    set outputs [torch::tensor_create {2.0 3.0 4.0} float64 cpu true]
    set inputs [torch::tensor_create {1.0 2.0 3.0} float64 cpu true]
    
    ;# Test gradient computation with float64
    set result [torch::grad -outputs $outputs -inputs $inputs]
    
    ;# Check that result is valid
    expr {$result ne "" && [string match "tensor*" $result]}
} {1}

;# Test 10: Edge case - single element tensors
test grad-10.1 {Single element tensors} {
    ;# Create single element tensors
    set outputs [torch::tensor_create {5.0} float32 cpu true]
    set inputs [torch::tensor_create {2.0} float32 cpu true]
    
    ;# Test gradient computation
    set result [torch::grad -outputs $outputs -inputs $inputs]
    
    ;# Check that result is valid
    expr {$result ne "" && [string match "tensor*" $result]}
} {1}

;# Cleanup
cleanupTests 