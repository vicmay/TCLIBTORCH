#!/usr/bin/env tclsh

# Test file for torch::meshgrid command with dual syntax support
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

# Test suite for torch::meshgrid
test meshgrid-1.1 {Basic positional syntax} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0} -dtype float32 -device cpu]
    
    # Test basic meshgrid
    set result [torch::meshgrid $tensor1 $tensor2]
    
    # Verify result is a valid list of tensor handles
    expr {[llength $result] == 2}
} {1}

test meshgrid-2.1 {Named parameter syntax - basic} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0} -dtype float32 -device cpu]
    
    # Test basic named parameter syntax
    set result [torch::meshgrid -tensors [list $tensor1 $tensor2]]
    
    # Verify result is a valid list of tensor handles
    expr {[llength $result] == 2}
} {1}

test meshgrid-3.1 {camelCase alias - positional syntax} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0} -dtype float32 -device cpu]
    
    # Test camelCase alias with positional syntax
    set result [torch::meshGrid $tensor1 $tensor2]
    
    # Verify result is a valid list of tensor handles
    expr {[llength $result] == 2}
} {1}

test meshgrid-3.2 {camelCase alias - named parameter syntax} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0} -dtype float32 -device cpu]
    
    # Test camelCase alias with named parameter syntax
    set result [torch::meshGrid -tensors [list $tensor1 $tensor2]]
    
    # Verify result is a valid list of tensor handles
    expr {[llength $result] == 2}
} {1}

test meshgrid-4.1 {Error handling - invalid tensor} {
    # Create one valid tensor
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    
    # Test with invalid tensor
    catch {torch::meshgrid $tensor1 "invalid_tensor"} err
    
    # Verify error message
    string match "*Invalid tensor*" $err
} {1}

test meshgrid-4.2 {Error handling - no tensors} {
    # Test with no tensors
    catch {torch::meshgrid} err
    
    # Verify error message
    string match "*At least one tensor is required*" $err
} {1}

test meshgrid-4.3 {Error handling - invalid parameter} {
    # Create tensor
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    
    # Test with invalid parameter
    catch {torch::meshgrid -invalid $tensor1} err
    
    # Verify error message
    string match "*Unknown parameter*" $err
} {1}

test meshgrid-5.1 {Three-dimensional grid} {
    # Create three tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {3.0 4.0 5.0} -dtype float32 -device cpu]
    set tensor3 [torch::tensor_create -data {6.0 7.0} -dtype float32 -device cpu]
    
    # Test with three tensors
    set result [torch::meshgrid $tensor1 $tensor2 $tensor3]
    
    # Verify result has three tensors
    expr {[llength $result] == 3}
} {1}

test meshgrid-6.1 {Syntax consistency - results match} {
    # Create two tensors
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set tensor2 [torch::tensor_create -data {4.0 5.0} -dtype float32 -device cpu]
    
    # Get results using both syntaxes
    set result1 [torch::meshgrid $tensor1 $tensor2]
    set result2 [torch::meshgrid -tensors [list $tensor1 $tensor2]]
    
    # Compare number of results
    expr {[llength $result1] == [llength $result2]}
} {1}

cleanupTests
