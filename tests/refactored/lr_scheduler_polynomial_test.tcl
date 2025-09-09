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
test lr_scheduler_polynomial-1.1 {Basic positional syntax with required parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial $optimizer 100]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-1.2 {Positional syntax with power parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial $optimizer 50 2.0]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-1.3 {Positional syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial $optimizer 200 1.5 10]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

# Test 2: Named parameter syntax
test lr_scheduler_polynomial-2.1 {Named syntax with required parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 100]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-2.2 {Named syntax with snake_case parameter alias} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -total_iters 150]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-2.3 {Named syntax with power parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 80 -power 0.5]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-2.4 {Named syntax with lastEpoch parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 60 -lastEpoch 5]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-2.5 {Named syntax with snake_case lastEpoch alias} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 60 -last_epoch 5]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-2.6 {Named syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 120 -power 3.0 -lastEpoch 20]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-2.7 {Named syntax with parameters in different order} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -power 2.5 -lastEpoch 15 -optimizer $optimizer -totalIters 90]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

# Test 3: CamelCase alias
test lr_scheduler_polynomial-3.1 {CamelCase alias with positional syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerPolynomial $optimizer 75 1.8 12]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-3.2 {CamelCase alias with named syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerPolynomial -optimizer $optimizer -totalIters 110 -power 2.2 -lastEpoch 8]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

# Test 4: Syntax consistency - both syntaxes should work identically
test lr_scheduler_polynomial-4.1 {Positional and named syntax consistency - basic} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_polynomial $optimizer1 100]
    set scheduler2 [torch::lr_scheduler_polynomial -optimizer $optimizer2 -totalIters 100]
    # Both should create valid scheduler handles
    set result [expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0}]
    cleanup_resources [list $optimizer1 $optimizer2 $scheduler1 $scheduler2]
    set result
} 1

test lr_scheduler_polynomial-4.2 {Positional and named syntax consistency - all parameters} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_polynomial $optimizer1 50 2.0 5]
    set scheduler2 [torch::lr_scheduler_polynomial -optimizer $optimizer2 -totalIters 50 -power 2.0 -lastEpoch 5]
    # Both should create valid scheduler handles
    set result [expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0}]
    cleanup_resources [list $optimizer1 $optimizer2 $scheduler1 $scheduler2]
    set result
} 1

# Test 5: Functionality tests
test lr_scheduler_polynomial-5.1 {Scheduler creation with small totalIters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 10]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-5.2 {Scheduler creation with large totalIters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 10000]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-5.3 {Scheduler creation with zero power} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 100 -power 0.0]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-5.4 {Scheduler creation with fractional power} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 100 -power 0.5]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-5.5 {Scheduler creation with high power} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 100 -power 5.0]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-5.6 {Scheduler creation with positive lastEpoch} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 100 -lastEpoch 25]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-5.7 {Scheduler creation with zero lastEpoch} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 100 -lastEpoch 0]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

# Test 6: Error handling
test lr_scheduler_polynomial-6.1 {Error - missing optimizer parameter} {
    set result [catch {torch::lr_scheduler_polynomial} error_msg]
    set result
} 1

test lr_scheduler_polynomial-6.2 {Error - missing totalIters parameter} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_polynomial -optimizer $optimizer} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_polynomial-6.3 {Error - invalid optimizer handle} {
    set result [catch {torch::lr_scheduler_polynomial "invalid_optimizer" 100} error_msg]
    set result
} 1

test lr_scheduler_polynomial-6.4 {Error - zero totalIters} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 0} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_polynomial-6.5 {Error - negative totalIters} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters -50} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_polynomial-6.6 {Error - negative power} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 100 -power -1.0} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_polynomial-6.7 {Error - unknown parameter} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 100 -unknown_param 123} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_polynomial-6.8 {Error - odd number of named parameters} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_polynomial-6.9 {Error - invalid totalIters type} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters "not_a_number"} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_polynomial-6.10 {Error - invalid power type} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 100 -power "not_a_number"} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

test lr_scheduler_polynomial-6.11 {Error - invalid lastEpoch type} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 100 -lastEpoch "not_a_number"} error_msg]
    cleanup_resources [list $optimizer]
    set result
} 1

# Test 7: Parameter validation edge cases
test lr_scheduler_polynomial-7.1 {Minimal valid totalIters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 1]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-7.2 {Very small power value} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 100 -power 0.001]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

test lr_scheduler_polynomial-7.3 {Negative lastEpoch (should be valid)} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 100 -lastEpoch -5]
    set result [expr {[string length $scheduler] > 0}]
    cleanup_resources [list $optimizer $scheduler]
    set result
} 1

cleanupTests 