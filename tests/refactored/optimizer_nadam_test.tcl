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
test optimizer_nadam-1.1 {Basic positional syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_nadam $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_nadam-1.2 {Positional syntax with learning rate} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_nadam $tensor 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_nadam-1.3 {Positional syntax with learning rate and betas} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_nadam $tensor 0.01 {0.8 0.9}]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_nadam-1.4 {Full positional syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_nadam $tensor 0.01 {0.8 0.9} 1e-8 0.01 0.004]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test cases for named parameter syntax
test optimizer_nadam-2.1 {Basic named parameter syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_nadam -parameters $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_nadam-2.2 {Named parameter syntax with learning rate} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_nadam -parameters $tensor -lr 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_nadam-2.3 {Named parameter syntax with betas} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_nadam -parameters $tensor -beta1 0.8 -beta2 0.9]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_nadam-2.4 {Full named parameter syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_nadam -parameters $tensor -lr 0.01 -beta1 0.8 -beta2 0.9 -eps 1e-8 -weightDecay 0.01 -momentumDecay 0.004]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test cases for camelCase alias
test optimizer_nadam-3.1 {CamelCase alias basic} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizerNadam -parameters $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_nadam-3.2 {CamelCase alias full parameters} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizerNadam -parameters $tensor -lr 0.01 -beta1 0.8 -beta2 0.9 -eps 1e-8 -weightDecay 0.01 -momentumDecay 0.004]
    expr {[string match "optimizer*" $opt]}
} {1}

# Error handling tests
test optimizer_nadam-4.1 {Error on missing parameters} -body {
    torch::optimizer_nadam
} -returnCodes error -result {Required parameters missing or invalid (parameters and positive values required for lr, valid beta values between 0-1, positive eps, non-negative weight decay and momentum decay)}

test optimizer_nadam-4.2 {Error on invalid parameter name} -body {
    set tensor [torch::zeros {5 5} float32]
    torch::optimizer_nadam -invalid $tensor
} -returnCodes error -result {Unknown parameter: -invalid}

test optimizer_nadam-4.3 {Error on missing parameter value} -body {
    set tensor [torch::zeros {5 5} float32]
    torch::optimizer_nadam -parameters
} -returnCodes error -result {Named parameters must come in pairs}

test optimizer_nadam-4.4 {Error on invalid learning rate} -body {
    set tensor [torch::zeros {5 5} float32]
    torch::optimizer_nadam -parameters $tensor -lr -1.0
} -returnCodes error -result {Required parameters missing or invalid (parameters and positive values required for lr, valid beta values between 0-1, positive eps, non-negative weight decay and momentum decay)}

# Functional tests
test optimizer_nadam-5.1 {Optimizer step} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_nadam -parameters $tensor]
    torch::optimizer_zero_grad $opt
    torch::optimizer_step $opt
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_nadam-5.2 {Multiple parameters} {
    set t1 [torch::zeros {5 5} float32]
    set t2 [torch::zeros {3 3} float32]
    set params [list $t1 $t2]
    set opt [torch::optimizer_nadam -parameters $params]
    torch::optimizer_zero_grad $opt
    torch::optimizer_step $opt
    expr {[string match "optimizer*" $opt]}
} {1}

cleanupTests 