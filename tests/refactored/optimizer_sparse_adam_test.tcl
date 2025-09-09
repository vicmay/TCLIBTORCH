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

# Helper function to create a sparse tensor
proc create_sparse_tensor {} {
    set tensor [torch::zeros {10 10} float32]
    set indices [torch::tensor_create -data {0 1 2 3 4} -dtype int64]
    set values [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32]
    return [list $tensor $indices $values]
}

# Test cases for positional syntax
test optimizer_sparse_adam-1.1 {Basic positional syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_sparse_adam $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_sparse_adam-1.2 {Positional syntax with learning rate} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_sparse_adam $tensor 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_sparse_adam-1.3 {Positional syntax with learning rate and betas} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_sparse_adam $tensor 0.01 0.8 0.9]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_sparse_adam-1.4 {Full positional syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_sparse_adam $tensor 0.01 0.8 0.9 1e-8 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test cases for named parameter syntax
test optimizer_sparse_adam-2.1 {Basic named parameter syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_sparse_adam -parameters $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_sparse_adam-2.2 {Named parameter syntax with learning rate} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_sparse_adam -parameters $tensor -lr 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_sparse_adam-2.3 {Named parameter syntax with betas} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_sparse_adam -parameters $tensor -beta1 0.8 -beta2 0.9]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_sparse_adam-2.4 {Full named parameter syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_sparse_adam -parameters $tensor -lr 0.01 -beta1 0.8 -beta2 0.9 -eps 1e-8 -weightDecay 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test cases for camelCase alias
test optimizer_sparse_adam-3.1 {CamelCase alias basic} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizerSparseAdam -parameters $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_sparse_adam-3.2 {CamelCase alias full parameters} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizerSparseAdam -parameters $tensor -lr 0.01 -beta1 0.8 -beta2 0.9 -eps 1e-8 -weightDecay 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

# Error handling tests
test optimizer_sparse_adam-4.1 {Error on missing parameters} -body {
    torch::optimizer_sparse_adam
} -returnCodes error -result {Usage: torch::optimizer_sparse_adam parameters ?lr? ?beta1? ?beta2? ?eps? ?weightDecay? | torch::optimizer_sparse_adam -parameters value ?-lr value? ?-beta1 value? ?-beta2 value? ?-eps value? ?-weightDecay value?}

test optimizer_sparse_adam-4.2 {Error on invalid parameter name} -body {
    set tensor [torch::zeros {5 5} float32]
    torch::optimizer_sparse_adam -invalid $tensor
} -returnCodes error -result {Unknown parameter: -invalid}

test optimizer_sparse_adam-4.3 {Error on missing parameter value} -body {
    set tensor [torch::zeros {5 5} float32]
    torch::optimizer_sparse_adam -parameters
} -returnCodes error -result {Missing value for parameter}

test optimizer_sparse_adam-4.4 {Error on invalid learning rate} -body {
    set tensor [torch::zeros {5 5} float32]
    torch::optimizer_sparse_adam -parameters $tensor -lr -1.0
} -returnCodes error -result {Required parameters missing or invalid (parameters and positive values required for lr, valid beta values between 0-1, positive eps, non-negative weight decay)}

# Functional tests
test optimizer_sparse_adam-5.1 {Optimizer step with sparse tensor} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_sparse_adam -parameters $tensor]
    torch::optimizer_zero_grad $opt
    torch::optimizer_step $opt
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_sparse_adam-5.2 {Multiple parameters} {
    set t1 [torch::zeros {5 5} float32]
    set t2 [torch::zeros {3 3} float32]
    set params [list $t1 $t2]
    set opt [torch::optimizer_sparse_adam -parameters $params]
    torch::optimizer_zero_grad $opt
    torch::optimizer_step $opt
    expr {[string match "optimizer*" $opt]}
} {1}

cleanupTests 