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

# Helper function to create SGD optimizer for testing
proc create_test_optimizer {} {
    set param_tensor [torch::tensor_create -data {0.5 0.3} -dtype float32]
    set param_list [list $param_tensor]
    return [torch::optimizer_sgd $param_list 0.01]
}

# Helper function to clean up tensors and optimizers
proc cleanup_resources {handles} {
    foreach handle $handles {
        catch {torch::tensor_delete $handle}
        catch {torch::optimizer_delete $handle}
        catch {torch::lr_scheduler_delete $handle}
    }
}

# Test 1: Basic positional syntax
test lr_scheduler_reduce_on_plateau-1.1 {Basic positional syntax with optimizer only} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau $optimizer]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-1.2 {Positional syntax with mode parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau $optimizer "max"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-1.3 {Positional syntax with mode and factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau $optimizer "min" 0.5]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-1.4 {Positional syntax with multiple parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau $optimizer "min" 0.2 20 1e-3]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-1.5 {Positional syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau $optimizer "max" 0.3 15 1e-5 "abs" 1e-6]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

# Test 2: Named parameter syntax
test lr_scheduler_reduce_on_plateau-2.1 {Named syntax with optimizer only} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-2.2 {Named syntax with mode parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -mode "max"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-2.3 {Named syntax with factor parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -factor 0.5]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-2.4 {Named syntax with patience parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -patience 25]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-2.5 {Named syntax with threshold parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -threshold 1e-3]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-2.6 {Named syntax with thresholdMode parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -thresholdMode "abs"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-2.7 {Named syntax with snake_case thresholdMode alias} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -threshold_mode "abs"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-2.8 {Named syntax with minLr parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -minLr 1e-6]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-2.9 {Named syntax with snake_case minLr alias} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -min_lr 1e-6]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-2.10 {Named syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau \
        -optimizer $optimizer \
        -mode "max" \
        -factor 0.4 \
        -patience 30 \
        -threshold 1e-2 \
        -thresholdMode "rel" \
        -minLr 1e-8]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-2.11 {Named syntax with parameters in different order} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau \
        -patience 12 \
        -factor 0.6 \
        -threshold 1e-5 \
        -optimizer $optimizer \
        -mode "min" \
        -minLr 1e-7 \
        -thresholdMode "abs"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

# Test 3: CamelCase alias
test lr_scheduler_reduce_on_plateau-3.1 {CamelCase alias with positional syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerReduceOnPlateau $optimizer "min" 0.4 15]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-3.2 {CamelCase alias with named syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerReduceOnPlateau \
        -optimizer $optimizer \
        -mode "max" \
        -factor 0.8 \
        -patience 20 \
        -threshold 1e-4]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

# Test 4: Syntax consistency - both syntaxes should work identically
test lr_scheduler_reduce_on_plateau-4.1 {Positional and named syntax consistency - defaults} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_reduce_on_plateau $optimizer1]
    set scheduler2 [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer2]
    # Both should create valid scheduler handles
    set result [expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0}]
    cleanup_resources [list $optimizer1 $optimizer2 $scheduler1 $scheduler2]
    set result
} 1

test lr_scheduler_reduce_on_plateau-4.2 {Positional and named syntax consistency - same parameters} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_reduce_on_plateau $optimizer1 "max" 0.25 30 1e-5 "abs" 1e-7]
    set scheduler2 [torch::lr_scheduler_reduce_on_plateau \
        -optimizer $optimizer2 \
        -mode "max" \
        -factor 0.25 \
        -patience 30 \
        -threshold 1e-5 \
        -thresholdMode "abs" \
        -minLr 1e-7]
    # Both should create valid scheduler handles
    set result [expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0}]
    cleanup_resources [list $optimizer1 $optimizer2 $scheduler1 $scheduler2]
    set result
} 1

# Test 5: Functionality tests
test lr_scheduler_reduce_on_plateau-5.1 {Scheduler creation with min mode} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -mode "min"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-5.2 {Scheduler creation with max mode} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -mode "max"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-5.3 {Scheduler creation with rel threshold mode} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -thresholdMode "rel"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-5.4 {Scheduler creation with abs threshold mode} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -thresholdMode "abs"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-5.5 {Scheduler creation with small factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -factor 0.01]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-5.6 {Scheduler creation with large patience} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -patience 100]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-5.7 {Scheduler creation with zero threshold} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -threshold 0.0]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-5.8 {Scheduler creation with zero minLr} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -minLr 0.0]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

# Test 6: Error handling
test lr_scheduler_reduce_on_plateau-6.1 {Error - missing optimizer parameter} {
    set result [catch {torch::lr_scheduler_reduce_on_plateau} error_msg]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.2 {Error - invalid optimizer handle} {
    set result [catch {torch::lr_scheduler_reduce_on_plateau "invalid_optimizer"} error_msg]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.3 {Error - invalid mode} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -mode "invalid"} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.4 {Error - negative factor} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -factor -0.1} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.5 {Error - factor greater than 1} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -factor 1.5} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.6 {Error - zero patience} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -patience 0} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.7 {Error - negative patience} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -patience -5} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.8 {Error - negative threshold} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -threshold -1e-4} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.9 {Error - negative minLr} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -minLr -1e-6} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.10 {Error - invalid thresholdMode} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -thresholdMode "invalid"} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.11 {Error - unknown parameter} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -unknown_param 123} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.12 {Error - odd number of named parameters} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -mode} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.13 {Error - invalid factor type} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -factor "not_a_number"} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.14 {Error - invalid patience type} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -patience "not_a_number"} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.15 {Error - invalid threshold type} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -threshold "not_a_number"} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_reduce_on_plateau-6.16 {Error - invalid minLr type} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -minLr "not_a_number"} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

# Test 7: Parameter validation edge cases
test lr_scheduler_reduce_on_plateau-7.1 {Edge case - factor exactly 1.0} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -factor 1.0]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-7.2 {Edge case - minimal valid patience} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -patience 1]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_reduce_on_plateau-7.3 {Edge case - very small factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer -factor 0.001]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

cleanupTests 