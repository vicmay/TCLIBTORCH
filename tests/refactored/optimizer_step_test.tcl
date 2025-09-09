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

# Helper function to create an optimizer
proc create_test_optimizer {} {
    set params [create_test_parameters]
    return [torch::optimizer_sgd $params 0.01]
}

# Test cases for positional syntax (backward compatibility)
test optimizer_step-1.1 {Basic positional syntax} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizer_step $optimizer]
    string equal $result "OK"
} {1}

test optimizer_step-1.2 {Multiple steps with same optimizer} {
    set optimizer [create_test_optimizer]
    set result1 [torch::optimizer_step $optimizer]
    set result2 [torch::optimizer_step $optimizer]
    expr {[string equal $result1 "OK"] && [string equal $result2 "OK"]}
} {1}

# Test cases for named parameter syntax (new)
test optimizer_step-2.1 {Named parameter syntax with -optimizer} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizer_step -optimizer $optimizer]
    string equal $result "OK"
} {1}

test optimizer_step-2.2 {Named parameter syntax with -opt} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizer_step -opt $optimizer]
    string equal $result "OK"
} {1}

# Test cases for camelCase alias
test optimizer_step-3.1 {CamelCase alias with positional syntax} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizerStep $optimizer]
    string equal $result "OK"
} {1}

test optimizer_step-3.2 {CamelCase alias with named parameter syntax} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizerStep -optimizer $optimizer]
    string equal $result "OK"
} {1}

# Error handling tests
test optimizer_step-4.1 {Error: missing parameters} {
    set result [catch {torch::optimizer_step} msg]
    expr {$result == 1 && [string match "*Required parameters missing*" $msg]}
} {1}

test optimizer_step-4.2 {Error: invalid optimizer handle} {
    set result [catch {torch::optimizer_step invalid_optimizer} msg]
    expr {$result == 1 && [string match "*Invalid optimizer handle*" $msg]}
} {1}

test optimizer_step-4.3 {Error: missing optimizer in named syntax} {
    set result [catch {torch::optimizer_step -lr 0.01} msg]
    expr {$result == 1 && [string match "*Unknown parameter*" $msg]}
} {1}

test optimizer_step-4.4 {Error: invalid parameter name} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::optimizer_step -invalid_param $optimizer} msg]
    expr {$result == 1 && [string match "*Unknown parameter*" $msg]}
} {1}

# Test consistency between syntaxes
test optimizer_step-5.1 {Syntax consistency: both produce same result} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set result1 [torch::optimizer_step $optimizer1]
    set result2 [torch::optimizer_step -optimizer $optimizer2]
    expr {[string equal $result1 "OK"] && [string equal $result2 "OK"]}
} {1}

test optimizer_step-5.2 {Works with different optimizer types} {
    set params [create_test_parameters]
    set sgd_opt [torch::optimizer_sgd $params 0.01]
    set adam_opt [torch::optimizer_adam $params 0.001]
    
    set result1 [torch::optimizer_step $sgd_opt]
    set result2 [torch::optimizer_step $adam_opt]
    expr {[string equal $result1 "OK"] && [string equal $result2 "OK"]}
} {1}

cleanupTests 