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

# Helper function to create test tensors
proc create_test_tensors {} {
    set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set tensor2 [torch::tensor_create -data {1.0 3.0 3.0} -dtype float32]
    return [list $tensor1 $tensor2]
}

# Test cases for positional syntax
test ne-1.1 {Basic positional syntax} {
    lassign [create_test_tensors] t1 t2
    set result [torch::ne $t1 $t2]
    set values [torch::tensor_to_list $result]
    set values
} {0 1 0}

test ne-1.2 {Positional syntax with different values} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set t2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32]
    set result [torch::ne $t1 $t2]
    set values [torch::tensor_to_list $result]
    set values
} {1 1 1}

test ne-1.3 {Positional syntax with identical values} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set t2 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set result [torch::ne $t1 $t2]
    set values [torch::tensor_to_list $result]
    set values
} {0 0 0}

# Test cases for named parameter syntax
test ne-2.1 {Named parameter syntax} {
    lassign [create_test_tensors] t1 t2
    set result [torch::ne -input1 $t1 -input2 $t2]
    set values [torch::tensor_to_list $result]
    set values
} {0 1 0}

test ne-2.2 {Named parameter syntax with tensor1/tensor2} {
    lassign [create_test_tensors] t1 t2
    set result [torch::ne -tensor1 $t1 -tensor2 $t2]
    set values [torch::tensor_to_list $result]
    set values
} {0 1 0}

# Test cases for camelCase alias
test ne-3.1 {CamelCase alias with positional syntax} {
    lassign [create_test_tensors] t1 t2
    set result [torch::Ne $t1 $t2]
    set values [torch::tensor_to_list $result]
    set values
} {0 1 0}

test ne-3.2 {CamelCase alias with named parameters} {
    lassign [create_test_tensors] t1 t2
    set result [torch::Ne -input1 $t1 -input2 $t2]
    set values [torch::tensor_to_list $result]
    set values
} {0 1 0}

# Error cases
test ne-4.1 {Error on missing arguments} {
    catch {torch::ne} err
    set err
} {Error in ne: Usage: torch::ne tensor1 tensor2 | torch::ne -input1 tensor1 -input2 tensor2}

test ne-4.2 {Error on invalid tensor} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    catch {torch::ne $t1 invalid_tensor} err
    set err
} {Error in ne: Invalid tensor name for input2}

test ne-4.3 {Error on invalid parameter name} {
    lassign [create_test_tensors] t1 t2
    catch {torch::ne -invalid $t1 -input2 $t2} err
    set err
} {Error in ne: Unknown parameter: -invalid. Valid parameters are: -input1, -tensor1, -input2, -tensor2}

cleanupTests
