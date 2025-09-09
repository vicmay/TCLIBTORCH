#!/usr/bin/env tclsh

package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl[info sharedlibextension]} err]} {
    puts "Failed to load libtorchtcl[info sharedlibextension]: $err"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper function to create test tensor
proc create_test_tensor {} {
    set tensor [torch::ones {4} float32 cpu]
    set values [torch::tensor_create -data {1.0 2.0 -1.0 -2.0} -dtype float32]
    set tensor [torch::tensor_mul $tensor $values]
    return $tensor
}

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

# Test cases for positional syntax
test silu-1.1 {Basic positional syntax} {
    set input [create_test_tensor]
    set result [torch::silu $input]
    set values [torch::tensor_to_list $result]
    # SiLU(x) = x * sigmoid(x), manually calculated values
    compare_tensor_values $values {0.7310585786300049 1.7615941559557649 -0.2689414213699951 -0.2384058440442351}
} {1}

test silu-1.2 {Error on wrong number of arguments} {
    catch {torch::silu} result
    set result
} {torch::silu: wrong # args: should be "torch::silu tensor"}

test silu-1.3 {Error on invalid tensor} {
    catch {torch::silu invalid_tensor} result
    set result
} {Invalid tensor name}

# Test cases for named parameter syntax
test silu-2.1 {Named parameter syntax} {
    set input [create_test_tensor]
    set result [torch::silu -input $input]
    set values [torch::tensor_to_list $result]
    # SiLU(x) = x * sigmoid(x), manually calculated values
    compare_tensor_values $values {0.7310585786300049 1.7615941559557649 -0.2689414213699951 -0.2384058440442351}
} {1}

test silu-2.2 {Error on missing input parameter} {
    catch {torch::silu -input} result
    set result
} {torch::silu: missing value for option -input}

test silu-2.3 {Error on unknown parameter} {
    catch {torch::silu -invalid tensor1} result
    set result
} {torch::silu: unknown option -invalid}

# Test cases for camelCase alias
test silu-3.1 {CamelCase alias basic usage} {
    set input [create_test_tensor]
    set result [torch::siLU -input $input]
    set values [torch::tensor_to_list $result]
    # SiLU(x) = x * sigmoid(x), manually calculated values
    compare_tensor_values $values {0.7310585786300049 1.7615941559557649 -0.2689414213699951 -0.2384058440442351}
} {1}

test silu-3.2 {CamelCase alias with positional syntax} {
    set input [create_test_tensor]
    set result [torch::siLU $input]
    set values [torch::tensor_to_list $result]
    # SiLU(x) = x * sigmoid(x), manually calculated values
    compare_tensor_values $values {0.7310585786300049 1.7615941559557649 -0.2689414213699951 -0.2384058440442351}
} {1}

cleanupTests 