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
test get_lr_advanced-1.1 {Basic positional syntax with step scheduler} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerSgd -parameters $params -lr 0.001]
    set scheduler [torch::lrSchedulerStep -optimizer $optimizer -stepSize 10 -gamma 0.1]
    set lr [torch::get_lr_advanced $scheduler]
    expr {$lr >= 0.0}
} {1}

test get_lr_advanced-1.2 {Basic positional syntax with exponential scheduler} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerAdam -parameters $params -lr 0.01]
    set scheduler [torch::lrSchedulerExponential -optimizer $optimizer -gamma 0.9]
    set lr [torch::get_lr_advanced $scheduler]
    expr {$lr >= 0.0}
} {1}

# Test cases for named parameter syntax
test get_lr_advanced-2.1 {Named parameter syntax with step scheduler} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerSgd -parameters $params -lr 0.005]
    set scheduler [torch::lrSchedulerStep -optimizer $optimizer -stepSize 5 -gamma 0.2]
    set lr [torch::get_lr_advanced -scheduler $scheduler]
    expr {$lr >= 0.0}
} {1}

test get_lr_advanced-2.2 {Named parameter syntax with cosine scheduler} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerAdam -parameters $params -lr 0.02]
    set scheduler [torch::lrSchedulerCosine -optimizer $optimizer -tMax 100]
    set lr [torch::get_lr_advanced -scheduler $scheduler]
    expr {$lr >= 0.0}
} {1}

# Test cases for camelCase alias
test get_lr_advanced-3.1 {CamelCase alias with positional syntax} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerSgd -parameters $params -lr 0.003]
    set scheduler [torch::lrSchedulerStep -optimizer $optimizer -stepSize 8 -gamma 0.5]
    set lr [torch::getLrAdvanced $scheduler]
    expr {$lr >= 0.0}
} {1}

test get_lr_advanced-3.2 {CamelCase alias with named parameters} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerAdam -parameters $params -lr 0.015]
    set scheduler [torch::lrSchedulerExponential -optimizer $optimizer -gamma 0.8]
    set lr [torch::getLrAdvanced -scheduler $scheduler]
    expr {$lr >= 0.0}
} {1}

# Test cases for error handling
test get_lr_advanced-4.1 {Invalid scheduler handle - positional} {
    set result [catch {torch::get_lr_advanced "invalid_scheduler"} msg]
    list $result [string match "*Invalid scheduler handle*" $msg]
} {1 1}

test get_lr_advanced-4.2 {Invalid scheduler handle - named parameters} {
    set result [catch {torch::get_lr_advanced -scheduler "invalid_scheduler"} msg]
    list $result [string match "*Invalid scheduler handle*" $msg]
} {1 1}

test get_lr_advanced-4.3 {Missing parameter value} {
    set result [catch {torch::get_lr_advanced -scheduler} msg]
    list $result [string match "*Missing value for parameter*" $msg]
} {1 1}

test get_lr_advanced-4.4 {Unknown parameter} {
    set result [catch {torch::get_lr_advanced -unknownParam value} msg]
    list $result [string match "*Unknown parameter*" $msg]
} {1 1}

test get_lr_advanced-4.5 {Missing required parameter} {
    set result [catch {torch::get_lr_advanced} msg]
    list $result [string match "*Required parameters missing*" $msg]
} {1 1}

# Test cases for syntax consistency
test get_lr_advanced-5.1 {Both syntaxes produce same result} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerSgd -parameters $params -lr 0.007]
    set scheduler [torch::lrSchedulerStep -optimizer $optimizer -stepSize 12 -gamma 0.3]
    set lr1 [torch::get_lr_advanced $scheduler]
    set lr2 [torch::get_lr_advanced -scheduler $scheduler]
    expr {abs($lr1 - $lr2) < 1e-10}
} {1}

test get_lr_advanced-5.2 {camelCase alias produces same result} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerAdam -parameters $params -lr 0.008]
    set scheduler [torch::lrSchedulerExponential -optimizer $optimizer -gamma 0.7]
    set lr1 [torch::get_lr_advanced $scheduler]
    set lr2 [torch::getLrAdvanced $scheduler]
    set lr3 [torch::getLrAdvanced -scheduler $scheduler]
    expr {abs($lr1 - $lr2) < 1e-10 && abs($lr1 - $lr3) < 1e-10}
} {1}

# Test cases for different scheduler types
test get_lr_advanced-6.1 {Multi-step scheduler learning rate} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerSgd -parameters $params -lr 0.004]
    set scheduler [torch::lrSchedulerMultiStep -optimizer $optimizer -milestones {30 60 90} -gamma 0.1]
    set lr [torch::get_lr_advanced -scheduler $scheduler]
    expr {$lr >= 0.0}
} {1}

test get_lr_advanced-6.2 {Cosine annealing scheduler learning rate} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerAdam -parameters $params -lr 0.006]
    set scheduler [torch::lrSchedulerCosineAnnealing -optimizer $optimizer -tMax 200]
    set lr [torch::get_lr_advanced -scheduler $scheduler]
    expr {$lr >= 0.0}
} {1}

test get_lr_advanced-6.3 {Plateau scheduler learning rate} {
    set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
    set optimizer [torch::optimizerSgd -parameters $params -lr 0.012]
    set scheduler [torch::lrSchedulerPlateau -optimizer $optimizer -mode min -factor 0.1 -patience 5]
    set lr [torch::get_lr_advanced -scheduler $scheduler]
    expr {$lr >= 0.0}
} {1}

cleanupTests 