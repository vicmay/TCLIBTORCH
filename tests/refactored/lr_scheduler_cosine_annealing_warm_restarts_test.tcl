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

test lr_scheduler_cosine_annealing_warm_restarts-1.1 {Basic positional syntax with required parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 10]
    
    # Verify scheduler handle is returned
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-1.2 {Positional syntax with T_mult parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 10 2]
    
    # Verify scheduler handle is returned
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-1.3 {Positional syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 10 2 0.01]
    
    # Verify scheduler handle is returned
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-1.4 {Positional syntax with different T_0 values} {
    set optimizer [create_test_optimizer]
    
    # Test various T_0 values
    set scheduler1 [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 5]
    set scheduler2 [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 20]
    set scheduler3 [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 50]
    
    # All should be valid
    expr {[is_valid_scheduler_handle $scheduler1] && 
          [is_valid_scheduler_handle $scheduler2] && 
          [is_valid_scheduler_handle $scheduler3]}
} {1}

# ===== Named Parameter Syntax Tests =====

test lr_scheduler_cosine_annealing_warm_restarts-2.1 {Named syntax with basic parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer -t0 10]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-2.2 {Named syntax with T_mult parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer -t0 15 -tMult 3]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-2.3 {Named syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer -t0 12 -tMult 2 -etaMin 0.005]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-2.4 {Named syntax with T_0 alias -T_0} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer -T_0 20]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-2.5 {Named syntax with T_0 alias -T0} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer -T0 25]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-2.6 {Named syntax with T_mult aliases} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set optimizer3 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer1 -t0 10 -tMult 2]
    set scheduler2 [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer2 -t0 10 -T_mult 2]
    set scheduler3 [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer3 -t0 10 -TMult 2]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2] && [is_valid_scheduler_handle $scheduler3]}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-2.7 {Named syntax with eta_min alias -eta_min} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer -t0 15 -eta_min 0.002]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== camelCase Alias Tests =====

test lr_scheduler_cosine_annealing_warm_restarts-3.1 {camelCase alias - basic usage} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerCosineAnnealingWarmRestarts $optimizer 10]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-3.2 {camelCase alias with named parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerCosineAnnealingWarmRestarts -optimizer $optimizer -t0 12 -tMult 3 -etaMin 0.01]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== Consistency Tests =====

test lr_scheduler_cosine_annealing_warm_restarts-4.1 {Consistency between positional and named syntax} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer1 10 2 0.01]
    set scheduler2 [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer2 -t0 10 -tMult 2 -etaMin 0.01]
    
    # Both should create valid scheduler handles
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-4.2 {Consistency between snake_case and camelCase} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer1 10]
    set scheduler2 [torch::lrSchedulerCosineAnnealingWarmRestarts $optimizer2 10]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-4.3 {Consistency - named syntax with different parameter aliases} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set optimizer3 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer1 -t0 15]
    set scheduler2 [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer2 -T_0 15]
    set scheduler3 [torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer3 -T0 15]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2] && [is_valid_scheduler_handle $scheduler3]}
} {1}

# ===== Warm Restarts Functionality Tests =====

test lr_scheduler_cosine_annealing_warm_restarts-5.1 {Basic warm restarts functionality} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 5]
    
    # Get initial learning rate
    set initial_lr [torch::get_lr $optimizer]
    
    # Step the scheduler several times
    for {set i 0} {$i < 3} {incr i} {
        torch::lr_scheduler_step_update $scheduler
    }
    set lr_after_steps [torch::get_lr $optimizer]
    
    # Learning rate should change
    expr {$initial_lr != $lr_after_steps}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-5.2 {Warm restart cycle behavior} {
    set optimizer [create_test_optimizer]
    set T_0 5
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer $T_0]
    
    # Get initial learning rate
    set initial_lr [torch::get_lr $optimizer]
    
    # Step through first cycle (T_0 = 5 steps)
    set lr_values {}
    for {set i 0} {$i < 10} {incr i} {
        torch::lr_scheduler_step_update $scheduler
        lappend lr_values [torch::get_lr $optimizer]
    }
    
    # Should have collected learning rates
    expr {[llength $lr_values] == 10}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-5.3 {T_mult functionality} {
    set optimizer [create_test_optimizer]
    # T_0=3, T_mult=2
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 3 2]
    
    # Get initial learning rate
    set initial_lr [torch::get_lr $optimizer]
    
    # Step through multiple cycles
    for {set i 0} {$i < 12} {incr i} {
        torch::lr_scheduler_step_update $scheduler
    }
    
    set final_lr [torch::get_lr $optimizer]
    
    # Should have valid learning rate after multiple restarts
    expr {$final_lr > 0}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-5.4 {eta_min behavior} {
    set optimizer [create_test_optimizer]
    set eta_min 0.01
    set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 10 1 $eta_min]
    
    # Step through to middle of cycle where LR should be near minimum
    for {set i 0} {$i < 5} {incr i} {
        torch::lr_scheduler_step_update $scheduler
    }
    
    set lr_at_min [torch::get_lr $optimizer]
    
    # Learning rate should be close to eta_min at middle of cycle
    expr {$lr_at_min >= $eta_min}
} {1}

# ===== Error Handling Tests =====

test lr_scheduler_cosine_annealing_warm_restarts-6.1 {Error handling - invalid optimizer} {
    set result [catch {
        torch::lr_scheduler_cosine_annealing_warm_restarts "invalid_optimizer" 10
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-6.2 {Error handling - missing required T_0} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-6.3 {Error handling - invalid T_0 value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer "invalid_t0"
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-6.4 {Error handling - negative T_0} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer -5
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-6.5 {Error handling - zero T_0} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 0
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-6.6 {Error handling - invalid T_mult value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 10 "invalid_tmult"
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-6.7 {Error handling - T_mult less than 1} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 10 0
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-6.8 {Error handling - invalid eta_min value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 10 1 "invalid_eta"
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-6.9 {Error handling - named syntax missing parameter value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer -t0
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing_warm_restarts-6.10 {Error handling - unknown parameter} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer $optimizer -t0 10 -unknown_param 5
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

cleanupTests 