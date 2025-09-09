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
test optimizer_adafactor-1.1 {Basic positional syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_adafactor $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_adafactor-1.2 {Positional syntax with learning rate} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_adafactor $tensor 0.8]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_adafactor-1.3 {Positional syntax with learning rate and eps2} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_adafactor $tensor 0.8 1e-30]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_adafactor-1.4 {Full positional syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_adafactor $tensor 0.8 1e-30 1.0 -1.0 -1.0 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test cases for named parameter syntax
test optimizer_adafactor-2.1 {Basic named parameter syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_adafactor -parameters $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_adafactor-2.2 {Named parameter syntax with learning rate} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_adafactor -parameters $tensor -lr 0.8]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_adafactor-2.3 {Named parameter syntax with eps2} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_adafactor -parameters $tensor -eps2 1e-30]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_adafactor-2.4 {Full named parameter syntax} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_adafactor -parameters $tensor -lr 0.8 -eps2 1e-30 -clipThreshold 1.0 -decayRate -1.0 -beta1 -1.0 -weightDecay 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test cases for camelCase alias
test optimizer_adafactor-3.1 {CamelCase alias basic} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizerAdafactor -parameters $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_adafactor-3.2 {CamelCase alias full parameters} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizerAdafactor -parameters $tensor -lr 0.8 -eps2 1e-30 -clipThreshold 1.0 -decayRate -1.0 -beta1 -1.0 -weightDecay 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

# Error handling tests
test optimizer_adafactor-4.1 {Error on missing parameters} -body {
    torch::optimizer_adafactor
} -returnCodes error -result {Required parameters missing}

test optimizer_adafactor-4.2 {Error on invalid parameter name} -body {
    set tensor [torch::zeros {5 5} float32]
    torch::optimizer_adafactor -invalid $tensor
} -returnCodes error -result {Unknown parameter: -invalid}

test optimizer_adafactor-4.3 {Error on missing parameter value} -body {
    set tensor [torch::zeros {5 5} float32]
    torch::optimizer_adafactor -parameters
} -returnCodes error -result {Missing value for parameter}

test optimizer_adafactor-4.4 {Error on invalid learning rate} -body {
    set tensor [torch::zeros {5 5} float32]
    torch::optimizer_adafactor -parameters $tensor -lr -1.0
} -returnCodes error -match glob -result {Invalid learning rate: -1*}

# Functional tests
test optimizer_adafactor-5.1 {Optimizer step} {
    set tensor [torch::zeros {5 5} float32]
    set opt [torch::optimizer_adafactor -parameters $tensor]
    torch::optimizer_zero_grad $opt
    torch::optimizer_step $opt
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_adafactor-5.2 {Multiple parameters} {
    set t1 [torch::zeros {5 5} float32]
    set t2 [torch::zeros {3 3} float32]
    set params [list $t1 $t2]
    set opt [torch::optimizer_adafactor -parameters $params]
    torch::optimizer_zero_grad $opt
    torch::optimizer_step $opt
    expr {[string match "optimizer*" $opt]}
} {1}

cleanupTests 