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
test get_lr-1.1 {Basic positional syntax with SGD optimizer} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerSgd -parameters $params -lr 0.001]
    set lr [torch::get_lr $optimizer]
    expr {abs($lr - 0.001) < 1e-6}
} {1}

test get_lr-1.2 {Basic positional syntax with Adam optimizer} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerAdam -parameters $params -lr 0.01]
    set lr [torch::get_lr $optimizer]
    expr {abs($lr - 0.01) < 1e-6}
} {1}

# Test cases for named parameter syntax
test get_lr-2.1 {Named parameter syntax with SGD optimizer} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerSgd -parameters $params -lr 0.005]
    set lr [torch::get_lr -optimizer $optimizer]
    expr {abs($lr - 0.005) < 1e-6}
} {1}

test get_lr-2.2 {Named parameter syntax with Adam optimizer} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerAdam -parameters $params -lr 0.02]
    set lr [torch::get_lr -optimizer $optimizer]
    expr {abs($lr - 0.02) < 1e-6}
} {1}

# Test cases for camelCase alias
test get_lr-3.1 {CamelCase alias with positional syntax} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerSgd -parameters $params -lr 0.003]
    set lr [torch::getLr $optimizer]
    expr {abs($lr - 0.003) < 1e-6}
} {1}

test get_lr-3.2 {CamelCase alias with named parameters} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerAdam -parameters $params -lr 0.015]
    set lr [torch::getLr -optimizer $optimizer]
    expr {abs($lr - 0.015) < 1e-6}
} {1}

# Test cases for error handling
test get_lr-4.1 {Invalid optimizer handle - positional} {
    set result [catch {torch::get_lr "invalid_optimizer"} msg]
    list $result [string match "*Invalid optimizer name*" $msg]
} {1 1}

test get_lr-4.2 {Invalid optimizer handle - named parameters} {
    set result [catch {torch::get_lr -optimizer "invalid_optimizer"} msg]
    list $result [string match "*Invalid optimizer name*" $msg]
} {1 1}

test get_lr-4.3 {Missing parameter value} {
    set result [catch {torch::get_lr -optimizer} msg]
    list $result [string match "*Missing value for parameter*" $msg]
} {1 1}

test get_lr-4.4 {Unknown parameter} {
    set result [catch {torch::get_lr -unknownParam value} msg]
    list $result [string match "*Unknown parameter*" $msg]
} {1 1}

test get_lr-4.5 {Missing required parameter} {
    set result [catch {torch::get_lr} msg]
    list $result [string match "*Required parameters missing*" $msg]
} {1 1}

# Test cases for syntax consistency
test get_lr-5.1 {Both syntaxes produce same result} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerSgd -parameters $params -lr 0.007]
    set lr1 [torch::get_lr $optimizer]
    set lr2 [torch::get_lr -optimizer $optimizer]
    expr {abs($lr1 - $lr2) < 1e-10}
} {1}

test get_lr-5.2 {camelCase alias produces same result} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerAdam -parameters $params -lr 0.008]
    set lr1 [torch::get_lr $optimizer]
    set lr2 [torch::getLr $optimizer]
    set lr3 [torch::getLr -optimizer $optimizer]
    expr {abs($lr1 - $lr2) < 1e-10 && abs($lr1 - $lr3) < 1e-10}
} {1}

# Test cases for different optimizer types
test get_lr-6.1 {AdamW optimizer learning rate} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerAdamW -parameters $params -lr 0.004]
    set lr [torch::get_lr -optimizer $optimizer]
    expr {abs($lr - 0.004) < 1e-6}
} {1}

test get_lr-6.2 {RMSprop optimizer learning rate} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerRmsprop -parameters $params -lr 0.006]
    set lr [torch::get_lr -optimizer $optimizer]
    expr {abs($lr - 0.006) < 1e-6}
} {1}

cleanupTests 