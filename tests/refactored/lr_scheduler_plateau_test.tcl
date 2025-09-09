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
test lr_scheduler_plateau-1.1 {Basic positional syntax with optimizer only} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau $optimizer]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-1.2 {Positional syntax with mode parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau $optimizer "max"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-1.3 {Positional syntax with mode and factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau $optimizer "min" 0.5]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-1.4 {Positional syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau $optimizer "min" 0.2 20]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

# Test 2: Named parameter syntax
test lr_scheduler_plateau-2.1 {Named syntax with optimizer only} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-2.2 {Named syntax with mode parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer -mode "max"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-2.3 {Named syntax with factor parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer -factor 0.5]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-2.4 {Named syntax with patience parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer -patience 15]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-2.5 {Named syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer -mode "max" -factor 0.3 -patience 25]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-2.6 {Named syntax with parameters in different order} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -patience 8 -factor 0.7 -optimizer $optimizer -mode "min"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

# Test 3: CamelCase alias
test lr_scheduler_plateau-3.1 {CamelCase alias with positional syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerPlateau $optimizer "min" 0.4 12]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-3.2 {CamelCase alias with named syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerPlateau -optimizer $optimizer -mode "max" -factor 0.6 -patience 18]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

# Test 4: Syntax consistency - both syntaxes should work identically
test lr_scheduler_plateau-4.1 {Positional and named syntax consistency - defaults} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_plateau $optimizer1]
    set scheduler2 [torch::lr_scheduler_plateau -optimizer $optimizer2]
    # Both should create valid scheduler handles
    set result [expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0}]
    cleanup_resources [list $optimizer1 $optimizer2 $scheduler1 $scheduler2]
    set result
} 1

test lr_scheduler_plateau-4.2 {Positional and named syntax consistency - same parameters} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_plateau $optimizer1 "max" 0.25 30]
    set scheduler2 [torch::lr_scheduler_plateau -optimizer $optimizer2 -mode "max" -factor 0.25 -patience 30]
    # Both should create valid scheduler handles
    set result [expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0}]
    cleanup_resources [list $optimizer1 $optimizer2 $scheduler1 $scheduler2]
    set result
} 1

# Test 5: Functionality tests
test lr_scheduler_plateau-5.1 {Scheduler creation with min mode} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer -mode "min"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-5.2 {Scheduler creation with max mode} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer -mode "max"]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-5.3 {Scheduler creation with small factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer -factor 0.1]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-5.4 {Scheduler creation with large patience} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer -patience 100]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-5.5 {Scheduler creation with edge case factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer -factor 1.0]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

# Test 6: Error handling
test lr_scheduler_plateau-6.1 {Error - missing optimizer parameter} {
    set result [catch {torch::lr_scheduler_plateau} error_msg]
    set result
} 1

test lr_scheduler_plateau-6.2 {Error - invalid optimizer handle} {
    set result [catch {torch::lr_scheduler_plateau "invalid_optimizer"} error_msg]
    set result
} 1

test lr_scheduler_plateau-6.3 {Error - invalid mode} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_plateau -optimizer $optimizer -mode "invalid"} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_plateau-6.4 {Error - negative factor} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_plateau -optimizer $optimizer -factor -0.1} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_plateau-6.5 {Error - factor greater than 1} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_plateau -optimizer $optimizer -factor 1.5} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_plateau-6.6 {Error - zero patience} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_plateau -optimizer $optimizer -patience 0} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_plateau-6.7 {Error - negative patience} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_plateau -optimizer $optimizer -patience -5} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_plateau-6.8 {Error - unknown parameter} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_plateau -optimizer $optimizer -unknown_param 123} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_plateau-6.9 {Error - odd number of named parameters} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_plateau -optimizer $optimizer -mode} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_plateau-6.10 {Error - invalid factor type} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_plateau -optimizer $optimizer -factor "not_a_number"} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_plateau-6.11 {Error - invalid patience type} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_plateau -optimizer $optimizer -patience "not_a_number"} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

# Test 7: Parameter validation edge cases
test lr_scheduler_plateau-7.1 {Minimal valid factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer -factor 0.001]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_plateau-7.2 {Minimal valid patience} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer -patience 1]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

cleanupTests 