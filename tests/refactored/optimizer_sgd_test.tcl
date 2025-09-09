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

# Helper function to create test tensors for parameters
proc create_test_parameters {} {
    set tensor1 [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set tensor2 [torch::tensor_create {4.0 5.0 6.0} float32 cpu true]
    return [list $tensor1 $tensor2]
}

# Test cases for positional syntax (backward compatibility)
test optimizer_sgd-1.1 {Basic positional syntax - minimal} {
    set params [create_test_parameters]
    set optimizer [torch::optimizer_sgd $params 0.01]
    expr {[string match "optimizer*" $optimizer]}
} {1}

test optimizer_sgd-1.2 {Positional syntax with momentum} {
    set params [create_test_parameters]
    set optimizer [torch::optimizer_sgd $params 0.01 0.9]
    expr {[string match "optimizer*" $optimizer]}
} {1}

test optimizer_sgd-1.3 {Positional syntax with momentum and dampening} {
    set params [create_test_parameters]
    set optimizer [torch::optimizer_sgd $params 0.01 0.9 0.1]
    expr {[string match "optimizer*" $optimizer]}
} {1}

test optimizer_sgd-1.4 {Positional syntax with weight decay} {
    set params [create_test_parameters]
    set optimizer [torch::optimizer_sgd $params 0.01 0.9 0.1 1e-4]
    expr {[string match "optimizer*" $optimizer]}
} {1}

test optimizer_sgd-1.5 {Positional syntax with Nesterov momentum} {
    set params [create_test_parameters]
    set optimizer [torch::optimizer_sgd $params 0.01 0.9 0.0 1e-4 true]
    expr {[string match "optimizer*" $optimizer]}
} {1}

# Test cases for named parameter syntax (new)
test optimizer_sgd-2.1 {Named parameter syntax - minimal} {
    set params [create_test_parameters]
    set optimizer [torch::optimizer_sgd -parameters $params -lr 0.01]
    expr {[string match "optimizer*" $optimizer]}
} {1}

test optimizer_sgd-2.2 {Named parameter syntax with momentum} {
    set params [create_test_parameters]
    set optimizer [torch::optimizer_sgd -parameters $params -lr 0.01 -momentum 0.9]
    expr {[string match "optimizer*" $optimizer]}
} {1}

test optimizer_sgd-2.3 {Named parameter syntax with all options} {
    set params [create_test_parameters]
    set optimizer [torch::optimizer_sgd -parameters $params -lr 0.01 -momentum 0.9 -dampening 0.0 -weightDecay 1e-4 -nesterov true]
    expr {[string match "optimizer*" $optimizer]}
} {1}

test optimizer_sgd-2.4 {Named parameter syntax with alternative parameter names} {
    set params [create_test_parameters]
    set optimizer [torch::optimizer_sgd -params $params -learningRate 0.01 -weight_decay 1e-4]
    expr {[string match "optimizer*" $optimizer]}
} {1}

# Test cases for camelCase alias
test optimizer_sgd-3.1 {CamelCase alias with positional syntax} {
    set params [create_test_parameters]
    set optimizer [torch::optimizerSgd $params 0.01]
    expr {[string match "optimizer*" $optimizer]}
} {1}

test optimizer_sgd-3.2 {CamelCase alias with named parameter syntax} {
    set params [create_test_parameters]
    set optimizer [torch::optimizerSgd -parameters $params -lr 0.01 -momentum 0.9]
    expr {[string match "optimizer*" $optimizer]}
} {1}

# Test parameter validation
test optimizer_sgd-4.1 {Parameter validation: missing parameters} {
    set result [catch {torch::optimizer_sgd -lr 0.01} msg]
    expr {$result == 1 && [string match "*Required parameters missing*" $msg]}
} {1}

test optimizer_sgd-4.2 {Parameter validation: invalid learning rate} {
    set params [create_test_parameters]
    set result [catch {torch::optimizer_sgd -parameters $params -lr -0.01} msg]
    expr {$result == 1 && [string match "*Required parameters missing*" $msg]}
} {1}

test optimizer_sgd-4.3 {Parameter validation: negative momentum} {
    set params [create_test_parameters]
    set result [catch {torch::optimizer_sgd -parameters $params -lr 0.01 -momentum -0.5} msg]
    expr {$result == 1 && [string match "*Required parameters missing*" $msg]}
} {1}

test optimizer_sgd-4.4 {Parameter validation: Nesterov without momentum} {
    set params [create_test_parameters]
    set result [catch {torch::optimizer_sgd -parameters $params -lr 0.01 -momentum 0.0 -nesterov true} msg]
    expr {$result == 1 && [string match "*Required parameters missing*" $msg]}
} {1}

test optimizer_sgd-4.5 {Parameter validation: Nesterov with dampening} {
    set params [create_test_parameters]
    set result [catch {torch::optimizer_sgd -parameters $params -lr 0.01 -momentum 0.9 -dampening 0.1 -nesterov true} msg]
    expr {$result == 1 && [string match "*Required parameters missing*" $msg]}
} {1}

# Test consistency between syntaxes
test optimizer_sgd-5.1 {Syntax consistency: both produce valid optimizers} {
    set params [create_test_parameters]
    set opt1 [torch::optimizer_sgd $params 0.01 0.9]
    set opt2 [torch::optimizer_sgd -parameters $params -lr 0.01 -momentum 0.9]
    expr {[string match "optimizer*" $opt1] && [string match "optimizer*" $opt2]}
} {1}

cleanupTests 