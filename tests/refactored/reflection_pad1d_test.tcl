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

# Helper function to check if two tensors are approximately equal
proc tensorsEqual {t1 t2} {
    set diff [torch::tensor_sub $t1 $t2]
    set max_diff [torch::tensor_max [torch::tensor_abs $diff]]
    set max_val [expr {[torch::tensor_item $max_diff] < 1e-5}]
    return $max_val
}

# Test cases for positional syntax
test reflection_pad1d-1.1 {Basic positional syntax - pad 1D tensor} {
    set input [torch::tensor_create -data {1.0 2.0 3.0} -shape {1 3} -dtype float32]
    set result [torch::reflection_pad1d $input {1 1}]
    set expected [torch::tensor_create -data {2.0 1.0 2.0 3.0 2.0} -shape {1 5} -dtype float32]
    tensorsEqual $result $expected
} {1}

test reflection_pad1d-1.2 {Positional syntax - error on wrong number of arguments} {
    catch {torch::reflection_pad1d} msg
    set msg
} {Usage: torch::reflection_pad1d tensor padding | torch::reflection_pad1d -input tensor -padding {left right}}

test reflection_pad1d-1.3 {Positional syntax - error on too many arguments} {
    catch {torch::reflection_pad1d tensor1 {1 1} extra} msg
    set msg
} {Usage: torch::reflection_pad1d tensor padding}

test reflection_pad1d-1.4 {Positional syntax - error on invalid padding format} {
    catch {torch::reflection_pad1d tensor1 {1}} msg
    set msg
} {Padding must be a list of 2 values for 1D}

# Test cases for named parameter syntax
test reflection_pad1d-2.1 {Named parameter syntax - pad 1D tensor} {
    set input [torch::tensor_create -data {1.0 2.0 3.0} -shape {1 3} -dtype float32]
    set result [torch::reflection_pad1d -input $input -padding {1 1}]
    set expected [torch::tensor_create -data {2.0 1.0 2.0 3.0 2.0} -shape {1 5} -dtype float32]
    tensorsEqual $result $expected
} {1}

test reflection_pad1d-2.2 {Named parameter syntax - error on missing value} {
    catch {torch::reflection_pad1d -input} msg
    set msg
} {Missing value for parameter}

test reflection_pad1d-2.3 {Named parameter syntax - error on unknown parameter} {
    catch {torch::reflection_pad1d -invalid tensor1 -padding {1 1}} msg
    set msg
} {Unknown parameter: -invalid}

test reflection_pad1d-2.4 {Named parameter syntax - error on missing input} {
    catch {torch::reflection_pad1d -padding {1 1}} msg
    set msg
} {Required parameters missing: input and/or padding}

test reflection_pad1d-2.5 {Named parameter syntax - error on missing padding} {
    set input [torch::tensor_create -data {1.0 2.0 3.0} -shape {1 3} -dtype float32]
    catch {torch::reflection_pad1d -input $input} msg
    set msg
} {Required parameters missing: input and/or padding}

# Test cases for camelCase alias
test reflection_pad1d-3.1 {CamelCase alias - pad 1D tensor} {
    set input [torch::tensor_create -data {1.0 2.0 3.0} -shape {1 3} -dtype float32]
    set result [torch::reflectionPad1d -input $input -padding {1 1}]
    set expected [torch::tensor_create -data {2.0 1.0 2.0 3.0 2.0} -shape {1 5} -dtype float32]
    tensorsEqual $result $expected
} {1}

# Test edge cases
test reflection_pad1d-4.1 {Edge case - zero padding} {
    set input [torch::tensor_create -data {1.0 2.0 3.0} -shape {1 3} -dtype float32]
    set result [torch::reflection_pad1d $input {0 0}]
    tensorsEqual $result $input
} {1}

test reflection_pad1d-4.2 {Edge case - single element tensor} {
    set input [torch::tensor_create -data {1.0} -shape {1 1} -dtype float32]
    set result [torch::reflection_pad1d $input {0 0}]
    tensorsEqual $result $input
} {1}

test reflection_pad1d-4.3 {Edge case - negative values} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0} -shape {1 3} -dtype float32]
    set result [torch::reflection_pad1d $input {1 1}]
    set expected [torch::tensor_create -data {-2.0 -1.0 -2.0 -3.0 -2.0} -shape {1 5} -dtype float32]
    tensorsEqual $result $expected
} {1}

test reflection_pad1d-4.4 {Edge case - large padding} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {1 4} -dtype float32]
    set result [torch::reflection_pad1d $input {2 2}]
    set expected [torch::tensor_create -data {3.0 2.0 1.0 2.0 3.0 4.0 3.0 2.0} -shape {1 8} -dtype float32]
    tensorsEqual $result $expected
} {1}

cleanupTests 