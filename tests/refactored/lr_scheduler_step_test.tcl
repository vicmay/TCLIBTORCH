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

# Test cases for torch::lr_scheduler_step

# Test 1: Basic positional syntax
test lr_scheduler_step-1.1 {Basic positional syntax} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.01]
    set result [torch::lr_scheduler_step $optimizer 5 0.1]
    string match "step_scheduler*" $result
} {1}

test lr_scheduler_step-1.2 {Positional syntax with default gamma} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.01]
    set result [torch::lr_scheduler_step $optimizer 10]
    string match "step_scheduler*" $result
} {1}

test lr_scheduler_step-1.3 {Positional syntax with custom gamma} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.01]
    set result [torch::lr_scheduler_step $optimizer 7 0.5]
    string match "step_scheduler*" $result
} {1}

# Test 2: Named parameter syntax
test lr_scheduler_step-2.1 {Named parameter syntax - minimal} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.01]
    set result [torch::lr_scheduler_step -optimizer $optimizer -stepSize 5]
    string match "step_scheduler*" $result
} {1}

test lr_scheduler_step-2.2 {Named parameter syntax - all parameters} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.01]
    set result [torch::lr_scheduler_step -optimizer $optimizer -stepSize 10 -gamma 0.5]
    string match "step_scheduler*" $result
} {1}

test lr_scheduler_step-2.3 {Named parameter syntax - parameter order independence} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.01]
    set result [torch::lr_scheduler_step -gamma 0.8 -stepSize 7 -optimizer $optimizer]
    string match "step_scheduler*" $result
} {1}

test lr_scheduler_step-2.4 {Named parameter syntax - step_size alias} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.01]
    set result [torch::lr_scheduler_step -optimizer $optimizer -step_size 5 -gamma 0.1]
    string match "step_scheduler*" $result
} {1}

# Test 3: camelCase alias
test lr_scheduler_step-3.1 {camelCase alias - positional syntax} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.01]
    set result [torch::lrSchedulerStep $optimizer 5 0.1]
    string match "step_scheduler*" $result
} {1}

test lr_scheduler_step-3.2 {camelCase alias - named syntax} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.01]
    set result [torch::lrSchedulerStep -optimizer $optimizer -stepSize 7 -gamma 0.3]
    string match "step_scheduler*" $result
} {1}

# Test 4: Syntax consistency
test lr_scheduler_step-4.1 {Syntax consistency - same parameters} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set opt1 [torch::optimizer_sgd [list $t1] 0.01]
    set opt2 [torch::optimizer_sgd [list $t1] 0.01]
    
    set result1 [torch::lr_scheduler_step $opt1 5 0.1]
    set result2 [torch::lr_scheduler_step -optimizer $opt2 -stepSize 5 -gamma 0.1]
    
    # Both should create valid scheduler handles
    expr {[string match "step_scheduler*" $result1] && [string match "step_scheduler*" $result2]}
} {1}

# Test 5: Functionality tests
test lr_scheduler_step-5.1 {Scheduler functionality - basic operation} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.1]
    set scheduler [torch::lr_scheduler_step $optimizer 2 0.5]
    
    # Get initial LR
    set initial_lr [torch::get_lr $optimizer]
    
    # Step the scheduler twice (should not change LR yet since step_size=2)
    torch::lr_scheduler_step_update $scheduler
    set lr_after_1 [torch::get_lr $optimizer]
    
    torch::lr_scheduler_step_update $scheduler
    set lr_after_2 [torch::get_lr $optimizer]
    
    # After 2 steps, LR should be reduced by gamma=0.5
    expr {abs($initial_lr - 0.1) < 1e-6 && abs($lr_after_1 - 0.1) < 1e-6 && abs($lr_after_2 - 0.05) < 1e-6}
} {1}

# Test 6: Error handling
test lr_scheduler_step-6.1 {Error handling - missing required parameters (positional)} {
    set result [catch {torch::lr_scheduler_step} error]
    list $result [string match "*Required parameters*" $error]
} {1 1}

test lr_scheduler_step-6.2 {Error handling - invalid optimizer handle} {
    set result [catch {torch::lr_scheduler_step invalid_optimizer 5 0.1} error]
    list $result [string match "*Invalid optimizer*" $error]
} {1 1}

test lr_scheduler_step-6.3 {Error handling - zero step_size} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.01]
    set result [catch {torch::lr_scheduler_step $optimizer 0 0.1} error]
    list $result [string match "*stepSize must be positive*" $error]
} {1 1}

test lr_scheduler_step-6.4 {Error handling - negative gamma} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.01]
    set result [catch {torch::lr_scheduler_step $optimizer 5 -0.1} error]
    list $result [string match "*gamma must be positive*" $error]
} {1 1}

test lr_scheduler_step-6.5 {Error handling - missing optimizer (named)} {
    set result [catch {torch::lr_scheduler_step -stepSize 5 -gamma 0.1} error]
    list $result [string match "*Required parameters*" $error]
} {1 1}

test lr_scheduler_step-6.6 {Error handling - unknown parameter} {
    set t1 [torch::tensor_create -data {2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
    set optimizer [torch::optimizer_sgd [list $t1] 0.01]
    set result [catch {torch::lr_scheduler_step -optimizer $optimizer -stepSize 5 -unknown_param value} error]
    list $result [string match "*Unknown parameter*" $error]
} {1 1}

cleanupTests
