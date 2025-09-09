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
test optimizer_novograd-1.1 {Basic positional syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_novograd $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_novograd-1.2 {Positional syntax with learning rate} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_novograd $tensor 0.001]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_novograd-1.3 {Positional syntax with betas} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_novograd $tensor 0.001 {0.95 0.98}]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_novograd-1.4 {Full positional syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_novograd $tensor 0.001 {0.95 0.98} 1e-8 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test cases for named parameter syntax
test optimizer_novograd-2.1 {Basic named parameter syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_novograd -parameters $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_novograd-2.2 {Named parameter syntax with learning rate} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_novograd -parameters $tensor -lr 0.001]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_novograd-2.3 {Named parameter syntax with betas} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_novograd -parameters $tensor -beta1 0.95 -beta2 0.98]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_novograd-2.4 {Full named parameter syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_novograd -parameters $tensor -lr 0.001 -beta1 0.95 -beta2 0.98 -eps 1e-8 -weightDecay 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test cases for camelCase alias
test optimizer_novograd-3.1 {CamelCase alias basic} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizerNovograd -parameters $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_novograd-3.2 {CamelCase alias full parameters} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizerNovograd -parameters $tensor -lr 0.001 -beta1 0.95 -beta2 0.98 -eps 1e-8 -weightDecay 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

# Error handling tests
test optimizer_novograd-4.1 {Error on missing parameters} -body {
    torch::optimizer_novograd
} -returnCodes error -result {Required parameters missing or invalid (parameters and positive values required for lr, valid beta values between 0-1, positive eps, non-negative weight decay)}

test optimizer_novograd-4.2 {Error on invalid parameter name} -body {
    set tensor [torch::zeros {5 5} float32]
    torch::optimizer_novograd -invalid $tensor
} -returnCodes error -result {Unknown parameter: -invalid}

test optimizer_novograd-4.3 {Error on missing parameter value} -body {
    set tensor [torch::zeros {5 5} float32]
    torch::optimizer_novograd -parameters
} -returnCodes error -result {Named parameters must come in pairs}

test optimizer_novograd-4.4 {Error on invalid learning rate} -body {
    set tensor [torch::zeros {5 5} float32]
    torch::optimizer_novograd -parameters $tensor -lr -1.0
} -returnCodes error -result {Required parameters missing or invalid (parameters and positive values required for lr, valid beta values between 0-1, positive eps, non-negative weight decay)}

# Functional tests
test optimizer_novograd-5.1 {Optimizer step} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_novograd -parameters $tensor]
    torch::optimizer_zero_grad $opt
    torch::optimizer_step $opt
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_novograd-5.2 {Multiple parameters} {
    set t1 [torch::zeros {5 5} float32]
    set t2 [torch::zeros {3 3} float32]
    set params [list $t1 $t2]
    set opt [torch::optimizer_novograd -parameters $params]
    torch::optimizer_zero_grad $opt
    torch::optimizer_step $opt
    expr {[string match "optimizer*" $opt]}
} {1}

cleanupTests
