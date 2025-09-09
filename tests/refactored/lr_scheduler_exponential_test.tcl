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

# Helper procedures
proc create_test_optimizer {} {
    # Create a simple model parameters for optimizer
    set weights [torch::tensor_create {1.0 2.0 3.0} float32]
    set optimizer [torch::optimizer_sgd $weights 0.1]
    return $optimizer
}

proc get_scheduler_handle {result} {
    # Extract scheduler handle from result
    return $result
}

proc is_valid_scheduler_handle {handle} {
    # Check if the handle looks like a scheduler handle
    return [string match "*scheduler*" $handle]
}

proc approximately_equal {val1 val2 {tolerance 1e-6}} {
    return [expr {abs($val1 - $val2) < $tolerance}]
}

# ===== Positional Syntax Tests =====

test lr_scheduler_exponential-1.1 {Basic positional syntax with default gamma} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential $optimizer]
    
    # Verify scheduler handle is returned
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_exponential-1.2 {Positional syntax with explicit gamma} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential $optimizer 0.9]
    
    # Verify scheduler handle is returned
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_exponential-1.3 {Positional syntax with different gamma values} {
    set optimizer [create_test_optimizer]
    
    # Test various gamma values
    set scheduler1 [torch::lr_scheduler_exponential $optimizer 0.8]
    set scheduler2 [torch::lr_scheduler_exponential $optimizer 0.99]
    set scheduler3 [torch::lr_scheduler_exponential $optimizer 0.5]
    
    # All should be valid
    expr {[is_valid_scheduler_handle $scheduler1] && 
          [is_valid_scheduler_handle $scheduler2] && 
          [is_valid_scheduler_handle $scheduler3]}
} {1}

# ===== Named Parameter Syntax Tests =====

test lr_scheduler_exponential-2.1 {Named syntax with -optimizer parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential -optimizer $optimizer]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_exponential-2.2 {Named syntax with -opt parameter alias} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential -opt $optimizer]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_exponential-2.3 {Named syntax with -gamma parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential -optimizer $optimizer -gamma 0.8]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_exponential-2.4 {Named syntax with -decay parameter alias} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential -optimizer $optimizer -decay 0.85]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_exponential-2.5 {Named syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential -optimizer $optimizer -gamma 0.75]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== camelCase Alias Tests =====

test lr_scheduler_exponential-3.1 {camelCase alias - basic usage} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerExponential $optimizer]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_exponential-3.2 {camelCase alias with named parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerExponential -optimizer $optimizer -gamma 0.9]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== Consistency Tests =====

test lr_scheduler_exponential-4.1 {Consistency between positional and named syntax} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_exponential $optimizer1 0.9]
    set scheduler2 [torch::lr_scheduler_exponential -optimizer $optimizer2 -gamma 0.9]
    
    # Both should create valid scheduler handles
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

test lr_scheduler_exponential-4.2 {Consistency between snake_case and camelCase} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_exponential $optimizer1]
    set scheduler2 [torch::lrSchedulerExponential $optimizer2]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

test lr_scheduler_exponential-4.3 {Consistency - named syntax with different parameter names} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_exponential -optimizer $optimizer1 -gamma 0.8]
    set scheduler2 [torch::lr_scheduler_exponential -opt $optimizer2 -decay 0.8]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

# ===== Learning Rate Decay Tests =====

test lr_scheduler_exponential-5.1 {Learning rate decay functionality} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential $optimizer 0.9]
    
    # Get initial learning rate
    set initial_lr [torch::get_lr $optimizer]
    
    # Step the scheduler
    torch::lr_scheduler_step_update $scheduler
    set lr_after_1_step [torch::get_lr $optimizer]
    
    # Step again
    torch::lr_scheduler_step_update $scheduler
    set lr_after_2_steps [torch::get_lr $optimizer]
    
    # Learning rate should decrease: initial_lr > lr_after_1_step > lr_after_2_steps
    expr {$initial_lr > $lr_after_1_step && $lr_after_1_step > $lr_after_2_steps}
} {1}

test lr_scheduler_exponential-5.2 {Exponential decay formula verification} {
    set optimizer [create_test_optimizer]
    set gamma 0.8
    set scheduler [torch::lr_scheduler_exponential $optimizer $gamma]
    
    # Get initial learning rate
    set initial_lr [torch::get_lr $optimizer]
    
    # Step once
    torch::lr_scheduler_step_update $scheduler
    set lr_after_1_step [torch::get_lr $optimizer]
    
    # Expected: lr = initial_lr * gamma^step_count
    set expected_lr [expr {$initial_lr * pow($gamma, 1)}]
    
    approximately_equal $lr_after_1_step $expected_lr
} {1}

test lr_scheduler_exponential-5.3 {Multiple steps exponential decay} {
    set optimizer [create_test_optimizer]
    set gamma 0.7
    set scheduler [torch::lr_scheduler_exponential $optimizer $gamma]
    
    set initial_lr [torch::get_lr $optimizer]
    
    # Step 3 times
    torch::lr_scheduler_step_update $scheduler
    torch::lr_scheduler_step_update $scheduler  
    torch::lr_scheduler_step_update $scheduler
    set lr_after_3_steps [torch::get_lr $optimizer]
    
    # Expected: lr = initial_lr * gamma^3
    set expected_lr [expr {$initial_lr * pow($gamma, 3)}]
    
    approximately_equal $lr_after_3_steps $expected_lr
} {1}

# ===== Different Gamma Values Tests =====

test lr_scheduler_exponential-6.1 {Conservative decay (gamma close to 1)} {
    set optimizer [create_test_optimizer]
    set gamma 0.99
    set scheduler [torch::lr_scheduler_exponential $optimizer $gamma]
    
    set initial_lr [torch::get_lr $optimizer]
    torch::lr_scheduler_step_update $scheduler
    set lr_after_step [torch::get_lr $optimizer]
    
    # Should decay very slowly
    set decay_ratio [expr {$lr_after_step / $initial_lr}]
    expr {$decay_ratio > 0.95 && $decay_ratio < 1.0}
} {1}

test lr_scheduler_exponential-6.2 {Aggressive decay (small gamma)} {
    set optimizer [create_test_optimizer]
    set gamma 0.5
    set scheduler [torch::lr_scheduler_exponential $optimizer $gamma]
    
    set initial_lr [torch::get_lr $optimizer]
    torch::lr_scheduler_step_update $scheduler
    set lr_after_step [torch::get_lr $optimizer]
    
    # Should decay significantly
    set decay_ratio [expr {$lr_after_step / $initial_lr}]
    approximately_equal $decay_ratio 0.5
} {1}

test lr_scheduler_exponential-6.3 {Default gamma behavior} {
    set optimizer [create_test_optimizer]
    # Uses default gamma = 0.95
    set scheduler [torch::lr_scheduler_exponential $optimizer]
    
    set initial_lr [torch::get_lr $optimizer]
    torch::lr_scheduler_step_update $scheduler
    set lr_after_step [torch::get_lr $optimizer]
    
    # Should match gamma = 0.95
    set expected_lr [expr {$initial_lr * 0.95}]
    approximately_equal $lr_after_step $expected_lr
} {1}

# ===== Error Handling Tests =====

test lr_scheduler_exponential-7.1 {Error - missing optimizer parameter} {
    set result [catch {torch::lr_scheduler_exponential} error]
    expr {$result == 1}
} {1}

test lr_scheduler_exponential-7.2 {Error - invalid optimizer handle} {
    set result [catch {torch::lr_scheduler_exponential invalid_optimizer} error]
    expr {$result == 1}
} {1}

test lr_scheduler_exponential-7.3 {Error - invalid gamma type} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_exponential $optimizer "invalid"} error]
    expr {$result == 1}
} {1}

test lr_scheduler_exponential-7.4 {Error - negative gamma} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_exponential $optimizer -0.1} error]
    expr {$result == 1}
} {1}

test lr_scheduler_exponential-7.5 {Error - zero gamma} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_exponential $optimizer 0.0} error]
    expr {$result == 1}
} {1}

test lr_scheduler_exponential-7.6 {Error - missing value for named parameter} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_exponential -optimizer} error]
    expr {$result == 1}
} {1}

test lr_scheduler_exponential-7.7 {Error - unknown parameter} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_exponential -invalid $optimizer} error]
    expr {$result == 1}
} {1}

test lr_scheduler_exponential-7.8 {Error - invalid gamma value in named syntax} {
    set optimizer [create_test_optimizer]
    set result [catch {torch::lr_scheduler_exponential -optimizer $optimizer -gamma "invalid"} error]
    expr {$result == 1}
} {1}

# ===== Integration Tests =====

test lr_scheduler_exponential-8.1 {Integration with different optimizers} {
    # Test with SGD
    set weights1 [torch::tensor_create {1.0 2.0} float32]
    set sgd_optimizer [torch::optimizer_sgd $weights1 0.01]
    set scheduler1 [torch::lr_scheduler_exponential $sgd_optimizer 0.9]
    
    # Test with Adam  
    set weights2 [torch::tensor_create {3.0 4.0} float32]
    set adam_optimizer [torch::optimizer_adam $weights2 0.001]
    set scheduler2 [torch::lr_scheduler_exponential $adam_optimizer 0.8]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

test lr_scheduler_exponential-8.2 {Multiple schedulers for same optimizer} {
    set optimizer [create_test_optimizer]
    
    # Create multiple schedulers (though typically only one would be used)
    set scheduler1 [torch::lr_scheduler_exponential $optimizer 0.9]
    set scheduler2 [torch::lr_scheduler_exponential $optimizer 0.8]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

# ===== Boundary Value Tests =====

test lr_scheduler_exponential-9.1 {Boundary - gamma very close to 1} {
    set optimizer [create_test_optimizer]
    set gamma 0.9999
    set scheduler [torch::lr_scheduler_exponential $optimizer $gamma]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_exponential-9.2 {Boundary - very small positive gamma} {
    set optimizer [create_test_optimizer]
    set gamma 0.001
    set scheduler [torch::lr_scheduler_exponential $optimizer $gamma]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_exponential-9.3 {Boundary - gamma equals 1 (no decay)} {
    set optimizer [create_test_optimizer]
    set gamma 1.0
    set scheduler [torch::lr_scheduler_exponential $optimizer $gamma]
    
    set initial_lr [torch::get_lr $optimizer]
    torch::lr_scheduler_step_update $scheduler
    set lr_after_step [torch::get_lr $optimizer]
    
    # With gamma = 1.0, learning rate should not change
    approximately_equal $initial_lr $lr_after_step
} {1}

# ===== Long Sequence Tests =====

test lr_scheduler_exponential-10.1 {Long decay sequence} {
    set optimizer [create_test_optimizer]
    set gamma 0.9
    set scheduler [torch::lr_scheduler_exponential $optimizer $gamma]
    
    set initial_lr [torch::get_lr $optimizer]
    
    # Step 10 times
    for {set i 0} {$i < 10} {incr i} {
        torch::lr_scheduler_step_update $scheduler
    }
    
    set final_lr [torch::get_lr $optimizer]
    set expected_lr [expr {$initial_lr * pow($gamma, 10)}]
    
    approximately_equal $final_lr $expected_lr
} {1}

test lr_scheduler_exponential-10.2 {Verify monotonic decrease} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential $optimizer 0.8]
    
    set prev_lr [torch::get_lr $optimizer]
    set monotonic 1
    
    # Check that learning rate decreases monotonically
    for {set i 0} {$i < 5} {incr i} {
        torch::lr_scheduler_step_update $scheduler
        set current_lr [torch::get_lr $optimizer]
        
        if {$current_lr >= $prev_lr} {
            set monotonic 0
            break
        }
        set prev_lr $current_lr
    }
    
    set monotonic
} {1}

cleanupTests 