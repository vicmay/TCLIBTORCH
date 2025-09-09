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

# Create test tensor
set tensor [torch::tensor_create -data {1.6 2.1 3.7 4.2 5.9 6.3} -shape {2 3} -dtype float32]

# Test cases for positional syntax
test round-1.1 {Basic positional syntax} {
    set result [torch::round $tensor]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {2 2 4 4 6 6}
} {1}

# Test cases for named parameter syntax
test round-2.1 {Named parameter syntax} {
    set result [torch::round -input $tensor]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {2 2 4 4 6 6}
} {1}

# Test cases for camelCase alias
test round-3.1 {CamelCase alias} {
    set result [torch::Round -input $tensor]
    set values [torch::tensor_to_list $result]
    compare_tensor_values $values {2 2 4 4 6 6}
} {1}

# Error handling tests
test round-4.1 {Error - invalid tensor} {
    catch {torch::round invalid_tensor} err
    set err
} {Invalid tensor name}

test round-4.2 {Error - missing input} {
    catch {torch::round} err
    set err
} {Usage: torch::round tensor | torch::round -input tensor}

cleanupTests 