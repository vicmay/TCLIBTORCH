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
test optimizer_rprop-1.1 {Basic positional syntax} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_rprop $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_rprop-1.2 {Positional syntax with learning rate} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_rprop $tensor 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_rprop-1.3 {Positional syntax with etas} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_rprop $tensor 0.01 {0.5 1.2}]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_rprop-1.4 {Full positional syntax} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_rprop $tensor 0.01 {0.5 1.2} {1e-6 50.0}]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test cases for named parameter syntax
test optimizer_rprop-2.1 {Basic named parameter syntax} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_rprop -parameters $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_rprop-2.2 {Named parameter syntax with learning rate} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_rprop -parameters $tensor -lr 0.01]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_rprop-2.3 {Named parameter syntax with etas} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_rprop -parameters $tensor -etas {0.5 1.2}]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_rprop-2.4 {Full named parameter syntax} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_rprop -parameters $tensor -lr 0.01 -etas {0.5 1.2} -stepSizes {1e-6 50.0}]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test cases for camelCase alias
test optimizer_rprop-3.1 {CamelCase alias basic} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizerRprop -parameters $tensor]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_rprop-3.2 {CamelCase alias full parameters} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizerRprop -parameters $tensor -lr 0.01 -etas {0.5 1.2} -stepSizes {1e-6 50.0}]
    expr {[string match "optimizer*" $opt]}
} {1}

# Error handling tests
test optimizer_rprop-4.1 {Error on missing parameters} -body {
    torch::optimizer_rprop
} -returnCodes error -result {Required parameters missing or invalid}

test optimizer_rprop-4.2 {Error on invalid parameter name} -body {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    torch::optimizer_rprop -invalid $tensor
} -returnCodes error -result {Unknown parameter: -invalid}

test optimizer_rprop-4.3 {Error on missing parameter value} -body {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    torch::optimizer_rprop -parameters
} -returnCodes error -result {Missing value for parameter}

test optimizer_rprop-4.4 {Error on invalid learning rate} -body {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    torch::optimizer_rprop -parameters $tensor -lr -1.0
} -returnCodes error -result {Required parameters missing or invalid}

# Functional tests
test optimizer_rprop-5.1 {Optimizer step} {
    set tensor [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set opt [torch::optimizer_rprop -parameters $tensor]
    torch::optimizer_zero_grad $opt
    torch::optimizer_step $opt
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_rprop-5.2 {Multiple parameters} {
    set t1 [torch::zeros -shape {5 5} -dtype float32 -device cpu]
    set t2 [torch::zeros -shape {3 3} -dtype float32 -device cpu]
    set params [list $t1 $t2]
    set opt [torch::optimizer_rprop -parameters $params]
    torch::optimizer_zero_grad $opt
    torch::optimizer_step $opt
    expr {[string match "optimizer*" $opt]}
} {1}

cleanupTests 