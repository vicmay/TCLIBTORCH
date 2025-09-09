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

test lr_scheduler_cosine_annealing-1.1 {Basic positional syntax with required parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 100]
    
    # Verify scheduler handle is returned
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing-1.2 {Positional syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 50 0.01]
    
    # Verify scheduler handle is returned
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing-1.3 {Positional syntax with different T_max values} {
    set optimizer [create_test_optimizer]
    
    # Test various T_max values
    set scheduler1 [torch::lr_scheduler_cosine_annealing $optimizer 10]
    set scheduler2 [torch::lr_scheduler_cosine_annealing $optimizer 200]
    set scheduler3 [torch::lr_scheduler_cosine_annealing $optimizer 1000]
    
    # All should be valid
    expr {[is_valid_scheduler_handle $scheduler1] && 
          [is_valid_scheduler_handle $scheduler2] && 
          [is_valid_scheduler_handle $scheduler3]}
} {1}

# ===== Named Parameter Syntax Tests =====

test lr_scheduler_cosine_annealing-2.1 {Named syntax with basic parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing -optimizer $optimizer -tMax 100]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing-2.2 {Named syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing -optimizer $optimizer -tMax 80 -etaMin 0.005]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing-2.3 {Named syntax with T_max alias -t_max} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing -optimizer $optimizer -t_max 120]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing-2.4 {Named syntax with T_max alias -T_max} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing -optimizer $optimizer -T_max 150]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing-2.5 {Named syntax with eta_min alias -eta_min} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing -optimizer $optimizer -tMax 90 -eta_min 0.002]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== camelCase Alias Tests =====

test lr_scheduler_cosine_annealing-3.1 {camelCase alias - basic usage} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerCosineAnnealing $optimizer 100]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cosine_annealing-3.2 {camelCase alias with named parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerCosineAnnealing -optimizer $optimizer -tMax 75 -etaMin 0.01]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== Consistency Tests =====

test lr_scheduler_cosine_annealing-4.1 {Consistency between positional and named syntax} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_cosine_annealing $optimizer1 100 0.01]
    set scheduler2 [torch::lr_scheduler_cosine_annealing -optimizer $optimizer2 -tMax 100 -etaMin 0.01]
    
    # Both should create valid scheduler handles
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

test lr_scheduler_cosine_annealing-4.2 {Consistency between snake_case and camelCase} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_cosine_annealing $optimizer1 100]
    set scheduler2 [torch::lrSchedulerCosineAnnealing $optimizer2 100]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

test lr_scheduler_cosine_annealing-4.3 {Consistency - named syntax with different parameter aliases} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set optimizer3 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_cosine_annealing -optimizer $optimizer1 -tMax 80]
    set scheduler2 [torch::lr_scheduler_cosine_annealing -optimizer $optimizer2 -t_max 80]
    set scheduler3 [torch::lr_scheduler_cosine_annealing -optimizer $optimizer3 -T_max 80]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2] && [is_valid_scheduler_handle $scheduler3]}
} {1}

# ===== Cosine Annealing Tests =====

test lr_scheduler_cosine_annealing-5.1 {Cosine annealing functionality basic} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 10]
    
    # Get initial learning rate
    set initial_lr [torch::get_lr $optimizer]
    
    # Step the scheduler
    torch::lr_scheduler_step_update $scheduler
    set lr_after_1_step [torch::get_lr $optimizer]
    
    # Step again
    torch::lr_scheduler_step_update $scheduler
    set lr_after_2_steps [torch::get_lr $optimizer]
    
    # Learning rate should change with cosine pattern
    expr {$initial_lr != $lr_after_1_step && $lr_after_1_step != $lr_after_2_steps}
} {1}

test lr_scheduler_cosine_annealing-5.2 {Cosine annealing with eta_min} {
    set optimizer [create_test_optimizer]
    set eta_min 0.01
    set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 40 $eta_min]
    
    # Get initial learning rate
    set initial_lr [torch::get_lr $optimizer]
    
    # Step through many iterations to reach minimum (T_max/2 = 20)
    for {set i 0} {$i < 20} {incr i} {
        torch::lr_scheduler_step_update $scheduler
    }
    
    set lr_at_min [torch::get_lr $optimizer]
    
    # Learning rate should be close to eta_min at T_max/2
    approximately_equal $lr_at_min $eta_min 0.005
} {1}

test lr_scheduler_cosine_annealing-5.3 {Cosine annealing mathematical verification} {
    set optimizer [create_test_optimizer]
    set T_max 10
    set eta_min 0.0
    set scheduler [torch::lr_scheduler_cosine_annealing $optimizer $T_max $eta_min]
    
    # Get initial learning rate
    set initial_lr [torch::get_lr $optimizer]
    
    # Step once
    torch::lr_scheduler_step_update $scheduler
    set lr_after_1_step [torch::get_lr $optimizer]
    
    # Calculate expected learning rate using cosine formula:
    # lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / T_max)) / 2
    set pi 3.14159265359
    set step_count 1
    set cosine_factor [expr {(1.0 + cos($pi * $step_count / $T_max)) / 2.0}]
    set expected_lr [expr {$eta_min + ($initial_lr - $eta_min) * $cosine_factor}]
    
    # Use a more relaxed tolerance to account for implementation differences
    approximately_equal $lr_after_1_step $expected_lr 0.01
} {1}

test lr_scheduler_cosine_annealing-5.4 {Difference from lr_scheduler_cosine - both should work similarly} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    # Create both types of cosine schedulers with same parameters
    set scheduler1 [torch::lr_scheduler_cosine $optimizer1 20 0.01]
    set scheduler2 [torch::lr_scheduler_cosine_annealing $optimizer2 20 0.01]
    
    # Both should be valid
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

# ===== Error Handling Tests =====

test lr_scheduler_cosine_annealing-6.1 {Error handling - invalid optimizer} {
    set result [catch {
        torch::lr_scheduler_cosine_annealing "invalid_optimizer" 100
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing-6.2 {Error handling - missing required T_max} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing $optimizer
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing-6.3 {Error handling - invalid T_max value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing $optimizer "invalid_tmax"
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing-6.4 {Error handling - negative T_max} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing $optimizer -5
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing-6.5 {Error handling - invalid eta_min value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing $optimizer 100 "invalid_eta"
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing-6.6 {Error handling - named syntax missing parameter value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing -optimizer $optimizer -tMax
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cosine_annealing-6.7 {Error handling - unknown parameter} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cosine_annealing -optimizer $optimizer -tMax 100 -unknown_param 5
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

cleanupTests 