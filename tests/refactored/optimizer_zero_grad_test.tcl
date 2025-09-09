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
test optimizer_zero_grad-1.1 {Basic positional syntax} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizer_zero_grad $optimizer]
    string equal $result "OK"
} {1}

test optimizer_zero_grad-1.2 {Positional syntax with set_to_none true} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizer_zero_grad $optimizer true]
    string equal $result "OK"
} {1}

test optimizer_zero_grad-1.3 {Positional syntax with set_to_none false} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizer_zero_grad $optimizer false]
    string equal $result "OK"
} {1}

test optimizer_zero_grad-1.4 {Multiple zero_grad calls} {
    set optimizer [create_test_optimizer]
    set result1 [torch::optimizer_zero_grad $optimizer]
    set result2 [torch::optimizer_zero_grad $optimizer]
    expr {[string equal $result1 "OK"] && [string equal $result2 "OK"]}
} {1}

# Test cases for named parameter syntax (new)
test optimizer_zero_grad-2.1 {Named parameter syntax with -optimizer} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizer_zero_grad -optimizer $optimizer]
    string equal $result "OK"
} {1}

test optimizer_zero_grad-2.2 {Named parameter syntax with -opt} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizer_zero_grad -opt $optimizer]
    string equal $result "OK"
} {1}

test optimizer_zero_grad-2.3 {Named parameter syntax with -setToNone true} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizer_zero_grad -optimizer $optimizer -setToNone true]
    string equal $result "OK"
} {1}

test optimizer_zero_grad-2.4 {Named parameter syntax with -setToNone false} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizer_zero_grad -optimizer $optimizer -setToNone false]
    string equal $result "OK"
} {1}

test optimizer_zero_grad-2.5 {Named parameter syntax with -set_to_none} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizer_zero_grad -optimizer $optimizer -set_to_none true]
    string equal $result "OK"
} {1}

# Test cases for camelCase alias
test optimizer_zero_grad-3.1 {CamelCase alias with positional syntax} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizerZeroGrad $optimizer]
    string equal $result "OK"
} {1}

test optimizer_zero_grad-3.2 {CamelCase alias with named parameter syntax} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizerZeroGrad -optimizer $optimizer -setToNone false]
    string equal $result "OK"
} {1}

test optimizer_zero_grad-3.3 {CamelCase alias with set_to_none parameter} {
    set optimizer [create_test_optimizer]
    set result [torch::optimizerZeroGrad $optimizer true]
    string equal $result "OK"
} {1}

# Error handling tests
test optimizer_zero_grad-4.1 {Error: missing parameters} {
    set result [catch {torch::optimizer_zero_grad} msg]
    expr {$result == 1 && [string match "*Required parameters missing*" $msg]}
} {1}

test optimizer_zero_grad-4.2 {Error: invalid optimizer handle} {
    set result [catch {torch::optimizer_zero_grad invalid_optimizer} msg]
    expr {$result == 1 && [string match "*Invalid optimizer handle*" $msg]}
} {1}

test optimizer_zero_grad-4.3 {Error: invalid set_to_none value} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::optimizer_zero_grad $optimizer invalid_bool} msg]
    expr {$result == 1 && [string match "*Invalid set_to_none*" $msg]}
} {1}

test optimizer_zero_grad-4.4 {Error: missing optimizer in named syntax} {
    set result [catch {torch::optimizer_zero_grad -setToNone true} msg]
    expr {$result == 1 && [string match "*Required parameters missing*" $msg]}
} {1}

test optimizer_zero_grad-4.5 {Error: invalid parameter name} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::optimizer_zero_grad -invalid_param $optimizer} msg]
    expr {$result == 1 && [string match "*Unknown parameter*" $msg]}
} {1}

test optimizer_zero_grad-4.6 {Error: invalid setToNone in named syntax} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::optimizer_zero_grad -optimizer $optimizer -setToNone invalid} msg]
    expr {$result == 1 && [string match "*Invalid set_to_none*" $msg]}
} {1}

# Test consistency between syntaxes
test optimizer_zero_grad-5.1 {Syntax consistency: both produce same result} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set result1 [torch::optimizer_zero_grad $optimizer1]
    set result2 [torch::optimizer_zero_grad -optimizer $optimizer2]
    expr {[string equal $result1 "OK"] && [string equal $result2 "OK"]}
} {1}

test optimizer_zero_grad-5.2 {Set to none parameter consistency} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set result1 [torch::optimizer_zero_grad $optimizer1 false]
    set result2 [torch::optimizer_zero_grad -optimizer $optimizer2 -setToNone false]
    expr {[string equal $result1 "OK"] && [string equal $result2 "OK"]}
} {1}

test optimizer_zero_grad-5.3 {Works with different optimizer types} {
    set params [create_test_parameters]
    set sgd_opt [torch::optimizer_sgd $params 0.01]
    set adam_opt [torch::optimizer_adam $params 0.001]
    
    set result1 [torch::optimizer_zero_grad $sgd_opt]
    set result2 [torch::optimizer_zero_grad $adam_opt]
    expr {[string equal $result1 "OK"] && [string equal $result2 "OK"]}
} {1}

cleanupTests 