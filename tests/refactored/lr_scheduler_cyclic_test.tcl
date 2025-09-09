#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper function to create a test optimizer
proc create_test_optimizer {} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} {3}]
    set optimizer [torch::optimizer_sgd $tensor 0.01]
    return $optimizer
}

# Helper function to check if a handle is a valid scheduler
proc is_valid_scheduler_handle {handle} {
    return [expr {[string match "*cyclic_scheduler*" $handle] && [string length $handle] > 0}]
}

# ===== Positional Syntax Tests =====

test lr_scheduler_cyclic-1.1 {Basic positional syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic $optimizer 0.001 0.1]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-1.2 {Positional syntax with step_size} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic $optimizer 0.005 0.05 1000]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-1.3 {Positional syntax with step_size and mode} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic $optimizer 0.001 0.1 1500 triangular2]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-1.4 {Positional syntax with exp_range mode} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic $optimizer 0.002 0.08 2000 exp_range]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== Named Parameter Syntax Tests =====

test lr_scheduler_cyclic-2.1 {Named syntax with required parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic -optimizer $optimizer -baseLr 0.001 -maxLr 0.1]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-2.2 {Named syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic -optimizer $optimizer -baseLr 0.005 -maxLr 0.05 -stepSize 1500 -mode triangular2]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-2.3 {Named syntax with base_lr alias} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic -optimizer $optimizer -base_lr 0.003 -maxLr 0.03]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-2.4 {Named syntax with max_lr alias} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic -optimizer $optimizer -baseLr 0.002 -max_lr 0.02]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-2.5 {Named syntax with step_size alias} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic -optimizer $optimizer -baseLr 0.001 -maxLr 0.1 -step_size 3000]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-2.6 {Named syntax with exp_range mode} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic -optimizer $optimizer -baseLr 0.001 -maxLr 0.1 -mode exp_range]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== camelCase Alias Tests =====

test lr_scheduler_cyclic-3.1 {camelCase alias - basic usage} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerCyclic $optimizer 0.001 0.1]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-3.2 {camelCase alias with named parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerCyclic -optimizer $optimizer -baseLr 0.005 -maxLr 0.05 -stepSize 2500 -mode triangular2]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== Consistency Tests =====

test lr_scheduler_cyclic-4.1 {Consistency between positional and named syntax} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_cyclic $optimizer1 0.001 0.1 2000 triangular]
    set scheduler2 [torch::lr_scheduler_cyclic -optimizer $optimizer2 -baseLr 0.001 -maxLr 0.1 -stepSize 2000 -mode triangular]
    
    # Both should create valid scheduler handles
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

test lr_scheduler_cyclic-4.2 {Consistency between snake_case and camelCase} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_cyclic $optimizer1 0.002 0.08]
    set scheduler2 [torch::lrSchedulerCyclic $optimizer2 0.002 0.08]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

test lr_scheduler_cyclic-4.3 {Consistency - named syntax with different parameter aliases} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_cyclic -optimizer $optimizer1 -baseLr 0.001 -maxLr 0.1]
    set scheduler2 [torch::lr_scheduler_cyclic -optimizer $optimizer2 -base_lr 0.001 -max_lr 0.1]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

# ===== Cyclic Scheduler Functionality Tests =====

test lr_scheduler_cyclic-5.1 {Default values functionality} {
    set optimizer [create_test_optimizer]
    # Default: step_size=2000, mode="triangular"
    set scheduler [torch::lr_scheduler_cyclic $optimizer 0.001 0.1]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-5.2 {Triangular mode functionality} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic $optimizer 0.001 0.1 1000 triangular]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-5.3 {Triangular2 mode functionality} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic $optimizer 0.001 0.1 1000 triangular2]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-5.4 {Exp_range mode functionality} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic $optimizer 0.001 0.1 1000 exp_range]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-5.5 {Large step size functionality} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic $optimizer 0.0001 0.01 10000]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_cyclic-5.6 {Small learning rate range} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_cyclic $optimizer 1e-6 1e-4]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== Error Handling Tests =====

test lr_scheduler_cyclic-6.1 {Error handling - invalid optimizer} {
    set result [catch {
        torch::lr_scheduler_cyclic "invalid_optimizer" 0.001 0.1
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cyclic-6.2 {Error handling - missing required baseLr} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cyclic $optimizer
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cyclic-6.3 {Error handling - missing required maxLr} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cyclic $optimizer 0.001
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cyclic-6.4 {Error handling - invalid baseLr value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cyclic $optimizer "invalid_base" 0.1
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cyclic-6.5 {Error handling - invalid maxLr value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cyclic $optimizer 0.001 "invalid_max"
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cyclic-6.6 {Error handling - negative baseLr} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cyclic $optimizer -0.001 0.1
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cyclic-6.7 {Error handling - maxLr <= baseLr} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cyclic $optimizer 0.1 0.05
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cyclic-6.8 {Error handling - invalid step_size value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cyclic $optimizer 0.001 0.1 "invalid_step"
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cyclic-6.9 {Error handling - zero step_size} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cyclic $optimizer 0.001 0.1 0
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cyclic-6.10 {Error handling - negative step_size} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cyclic $optimizer 0.001 0.1 -100
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cyclic-6.11 {Error handling - invalid mode} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cyclic $optimizer 0.001 0.1 1000 invalid_mode
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cyclic-6.12 {Error handling - named syntax missing parameter value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cyclic -optimizer $optimizer -baseLr
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_cyclic-6.13 {Error handling - unknown parameter} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_cyclic -optimizer $optimizer -baseLr 0.001 -maxLr 0.1 -unknown_param 5
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

cleanupTests 