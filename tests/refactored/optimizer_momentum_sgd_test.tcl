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

# Helper procedure to create simple parameter tensors
proc createParams {} {
    set w [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -shape {4 4} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {17.0 18.0 19.0 20.0} -shape {4} -dtype float32 -device cpu]
    return [list $w $b]
}

# Test 1: Positional syntax

test optimizer_momentum_sgd-1.1 {Positional syntax basic} {
    set params [createParams]
    set opt [torch::optimizer_momentum_sgd $params 0.1 0.9]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_momentum_sgd-1.2 {Positional syntax with weight decay} {
    set params [createParams]
    set opt [torch::optimizer_momentum_sgd $params 0.01 0.8 0.0005]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test 2: Named parameter syntax

test optimizer_momentum_sgd-2.1 {Named syntax basic} {
    set params [createParams]
    set opt [torch::optimizer_momentum_sgd -parameters $params -lr 0.05 -momentum 0.9]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_momentum_sgd-2.2 {Named syntax with weight decay} {
    set params [createParams]
    set opt [torch::optimizer_momentum_sgd -parameters $params -lr 0.05 -momentum 0.9 -weightDecay 0.001]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test 3: CamelCase alias

test optimizer_momentum_sgd-3.1 {CamelCase alias positional} {
    set params [createParams]
    set opt [torch::optimizerMomentumSgd $params 0.1 0.9]
    expr {[string match "optimizer*" $opt]}
} {1}

test optimizer_momentum_sgd-3.2 {CamelCase alias named} {
    set params [createParams]
    set opt [torch::optimizerMomentumSgd -parameters $params -lr 0.08 -momentum 0.7]
    expr {[string match "optimizer*" $opt]}
} {1}

# Test 4: Error handling

test optimizer_momentum_sgd-4.1 {Missing arguments positional} {
    set params [createParams]
    catch {torch::optimizer_momentum_sgd $params 0.1} result
    expr {[string match "*Usage: torch::optimizer_momentum_sgd*" $result]}
} {1}

test optimizer_momentum_sgd-4.2 {Invalid learning rate named} {
    set params [createParams]
    catch {torch::optimizer_momentum_sgd -parameters $params -lr -0.1 -momentum 0.9} result
    expr {[string match "*Invalid learning rate*" $result] || [string match "*Required parameters missing*" $result]}
} {1}

cleanupTests 