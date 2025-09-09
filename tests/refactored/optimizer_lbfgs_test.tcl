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

# Test cases for positional syntax
test optimizer_lbfgs-1.1 {Basic positional syntax} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_lbfgs $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_lbfgs-1.2 {Positional syntax with learning rate} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_lbfgs $tensor 1.0]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_lbfgs-1.3 {Positional syntax with max iterations} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_lbfgs $tensor 1.0 20]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_lbfgs-1.4 {Full positional syntax} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_lbfgs $tensor 1.0 20 25 1e-7 1e-9]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test cases for named parameter syntax
test optimizer_lbfgs-2.1 {Basic named parameter syntax} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_lbfgs -parameters $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_lbfgs-2.2 {Named parameter syntax with learning rate} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_lbfgs -parameters $tensor -lr 1.0]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_lbfgs-2.3 {Named parameter syntax with max iterations} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_lbfgs -parameters $tensor -maxIter 20]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_lbfgs-2.4 {Full named parameter syntax} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_lbfgs -parameters $tensor -lr 1.0 -maxIter 20 -maxEval 25 -toleranceGrad 1e-7 -toleranceChange 1e-9]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test cases for camelCase alias
test optimizer_lbfgs-3.1 {CamelCase alias basic} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizerLbfgs -parameters $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_lbfgs-3.2 {CamelCase alias full parameters} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizerLbfgs -parameters $tensor -lr 1.0 -maxIter 20 -maxEval 25 -toleranceGrad 1e-7 -toleranceChange 1e-9]
    expr {[string match "optimizer*" $opt]}
} {1}

# Error handling tests
test optimizer_lbfgs-4.1 {Error on missing parameters} -body {
    torch::optimizer_lbfgs
} -returnCodes error -result {Required parameters missing or invalid (parameters and positive values required for lr, maxIter, maxEval, toleranceGrad, toleranceChange)}

test optimizer_lbfgs-4.2 {Error on invalid parameter name} -body {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    torch::optimizer_lbfgs -invalid $tensor
} -returnCodes error -result {Unknown parameter: -invalid}

test optimizer_lbfgs-4.3 {Error on missing parameter value} -body {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    torch::optimizer_lbfgs -parameters
} -returnCodes error -result {Named parameters must come in pairs}

test optimizer_lbfgs-4.4 {Error on invalid learning rate} -body {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    torch::optimizer_lbfgs -parameters $tensor -lr -1.0
} -returnCodes error -result {Required parameters missing or invalid (parameters and positive values required for lr, maxIter, maxEval, toleranceGrad, toleranceChange)}

# Functional tests
test optimizer_lbfgs-5.1 {Optimizer creation} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_lbfgs -parameters $tensor]
    torch::optimizer_zero_grad $opt
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_lbfgs-5.2 {Multiple parameters} {
    set t1 [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set t2 [torch::zeros -shape {3 3} -dtype float32 -device cpu]
    set params [list $t1 $t2]
    set opt [torch::optimizer_lbfgs -parameters $params]
    torch::optimizer_zero_grad $opt
    expr {[string match "optimizer*" $opt]}
} {1}

cleanupTests 