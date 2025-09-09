#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper function to compare tensor values
proc compare_tensor_values {values expected} {
    if {[llength $values] != [llength $expected]} {
        return 0
    }
    foreach v $values e $expected {
        if {abs($v - $e) > 1e-6} {
            return 0
        }
    }
    return 1
}

# Create test tensors
set tensor1 [torch::tensor_create -data {1 2 3} -shape {1 3} -dtype float32]
set tensor2 [torch::tensor_create -data {4 5 6} -shape {1 3} -dtype float32]

# Test cases for positional syntax with list argument
test row_stack-1.1 {Basic positional syntax with list argument} {
    set result [torch::row_stack [list $tensor1 $tensor2]]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {1 2 3 4 5 6}
} {1}

# Test cases for positional syntax with multiple arguments
test row_stack-1.2 {Basic positional syntax with multiple arguments} {
    set result [torch::row_stack $tensor1 $tensor2]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {1 2 3 4 5 6}
} {1}

# Test cases for named parameter syntax
test row_stack-2.1 {Named parameter syntax with -tensors} {
    set result [torch::row_stack -tensors [list $tensor1 $tensor2]]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {1 2 3 4 5 6}
} {1}

test row_stack-2.2 {Named parameter syntax with -inputs} {
    set result [torch::row_stack -inputs [list $tensor1 $tensor2]]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {1 2 3 4 5 6}
} {1}

# Test cases for camelCase alias
test row_stack-3.1 {CamelCase alias with list argument} {
    set result [torch::rowStack -tensors [list $tensor1 $tensor2]]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {1 2 3 4 5 6}
} {1}

# Error handling tests
test row_stack-4.1 {Error - no arguments} {
    catch {torch::row_stack} err
    set err
} {wrong # args: should be "torch::row_stack tensor_list"}

test row_stack-4.2 {Error - invalid tensor} {
    catch {torch::row_stack invalid_tensor} err
    set err
} {Error in row_stack: Invalid tensor name}

test row_stack-4.3 {Error - unknown parameter} {
    catch {torch::row_stack -invalid $tensor1} err
    set err
} {Error in row_stack: Unknown parameter: -invalid}

cleanupTests 