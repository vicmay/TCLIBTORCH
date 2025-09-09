#!/usr/bin/env tclsh

# Test file for torch::narrow_copy command with dual syntax support
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

# Test suite for torch::narrow_copy
test narrow_copy-1.1 {Basic positional syntax} {
    # Create a test tensor
    set input [torch::arange 10]
    
    # Test narrow_copy with positional syntax
    set result [torch::narrow_copy $input 0 4 3]
    
    # Verify the result exists and is a tensor
    expr {[string match "tensor*" $result]}
} {1}

test narrow_copy-1.2 {Basic positional syntax - different range} {
    # Create a test tensor
    set input [torch::arange 10]
    
    # Test narrow_copy with positional syntax - different range
    set result [torch::narrow_copy $input 0 2 5]
    
    # Verify the result exists and is a tensor
    expr {[string match "tensor*" $result]}
} {1}

test narrow_copy-2.1 {Named parameter syntax} {
    # Create a test tensor
    set input [torch::arange 10]
    
    # Test narrow_copy with named parameter syntax
    set result [torch::narrow_copy -input $input -dim 0 -start 4 -length 3]
    
    # Verify the result exists and is a tensor
    expr {[string match "tensor*" $result]}
} {1}

test narrow_copy-2.2 {Named parameter syntax - different range} {
    # Create a test tensor
    set input [torch::arange 10]
    
    # Test narrow_copy with named parameter syntax - different range
    set result [torch::narrow_copy -input $input -dim 0 -start 2 -length 5]
    
    # Verify the result exists and is a tensor
    expr {[string match "tensor*" $result]}
} {1}

test narrow_copy-3.1 {CamelCase alias} {
    # Create a test tensor
    set input [torch::arange 10]
    
    # Test camelCase alias
    set result [torch::narrowCopy -input $input -dim 0 -start 4 -length 3]
    
    # Verify the result exists and is a tensor
    expr {[string match "tensor*" $result]}
} {1}

test narrow_copy-4.1 {Error handling - invalid input tensor} {
    # Test with invalid input tensor
    catch {torch::narrow_copy -input "invalid_tensor" -dim 0 -start 0 -length 1} err
    
    # Verify error message
    string match "*Invalid input tensor*" $err
} {1}

test narrow_copy-4.2 {Error handling - missing parameter} {
    # Create a test tensor
    set input [torch::arange 10]
    
    # Test with missing parameter
    catch {torch::narrow_copy -input $input -dim 0 -start} err
    
    # Verify error message
    string match "*Missing value for parameter*" $err
} {1}

test narrow_copy-4.3 {Error handling - invalid parameter} {
    # Create a test tensor
    set input [torch::arange 10]
    
    # Test with invalid parameter
    catch {torch::narrow_copy -input $input -invalid 0} err
    
    # Verify error message
    string match "*Unknown parameter*" $err
} {1}

cleanupTests
