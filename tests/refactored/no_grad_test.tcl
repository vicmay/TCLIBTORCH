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

# Helper function to create a test tensor
proc create_test_tensor {} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    return $tensor
}

# Test cases for no_grad
test no_grad-1.1 {Basic no_grad functionality} {
    set tensor [create_test_tensor]
    torch::no_grad
    set is_enabled [torch::is_grad_enabled]
    expr {!$is_enabled}
} {1}

test no_grad-1.2 {No grad prevents gradient computation} {
    set tensor [create_test_tensor]
    torch::no_grad
    set result [torch::tensor_mul $tensor 2.0]
    set requires_grad [torch::tensor_requires_grad $result]
    expr {!$requires_grad}
} {1}

test no_grad-1.3 {Restore grad mode after re-enabling} {
    set tensor [create_test_tensor]
    torch::no_grad
    torch::enable_grad
    set is_enabled [torch::is_grad_enabled]
    expr {$is_enabled}
} {1}

test no_grad-1.4 {camelCase alias} {
    set tensor [create_test_tensor]
    torch::noGrad
    set is_enabled [torch::is_grad_enabled]
    expr {!$is_enabled}
} {1}

test no_grad-2.1 {Error on extra arguments} {
    catch {torch::no_grad extra_arg} err
    set err
} {Error in no_grad: Usage: torch::no_grad}

cleanupTests 