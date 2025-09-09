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
    return [expr {[string match "*lambda_scheduler*" $handle] && [string length $handle] > 0}]
}

# ===== Positional Syntax Tests =====

test lr_scheduler_lambda-1.1 {Basic positional syntax without multiplier} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda $optimizer]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-1.2 {Positional syntax with multiplier} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda $optimizer 0.95]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-1.3 {Positional syntax with different multipliers} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set optimizer3 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_lambda $optimizer1 0.9]
    set scheduler2 [torch::lr_scheduler_lambda $optimizer2 0.8]
    set scheduler3 [torch::lr_scheduler_lambda $optimizer3 1.1]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2] && [is_valid_scheduler_handle $scheduler3]}
} {1}

test lr_scheduler_lambda-1.4 {Positional syntax with fractional multiplier} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda $optimizer 0.75]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== Named Parameter Syntax Tests =====

test lr_scheduler_lambda-2.1 {Named syntax with optimizer only} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda -optimizer $optimizer]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-2.2 {Named syntax with multiplier parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda -optimizer $optimizer -multiplier 0.9]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-2.3 {Named syntax with lambda alias} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda -optimizer $optimizer -lambda 0.85]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-2.4 {Named syntax with different multiplier values} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set optimizer3 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_lambda -optimizer $optimizer1 -multiplier 0.5]
    set scheduler2 [torch::lr_scheduler_lambda -optimizer $optimizer2 -lambda 1.5]
    set scheduler3 [torch::lr_scheduler_lambda -optimizer $optimizer3 -multiplier 2.0]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2] && [is_valid_scheduler_handle $scheduler3]}
} {1}

# ===== camelCase Alias Tests =====

test lr_scheduler_lambda-3.1 {camelCase alias - basic usage} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerLambda $optimizer]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-3.2 {camelCase alias with positional multiplier} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerLambda $optimizer 0.88]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-3.3 {camelCase alias with named parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerLambda -optimizer $optimizer -multiplier 0.92]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-3.4 {camelCase alias with lambda parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerLambda -optimizer $optimizer -lambda 0.78]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== Consistency Tests =====

test lr_scheduler_lambda-4.1 {Consistency between positional and named syntax} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_lambda $optimizer1 0.9]
    set scheduler2 [torch::lr_scheduler_lambda -optimizer $optimizer2 -multiplier 0.9]
    
    # Both should create valid scheduler handles
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

test lr_scheduler_lambda-4.2 {Consistency between snake_case and camelCase} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_lambda $optimizer1 0.85]
    set scheduler2 [torch::lrSchedulerLambda $optimizer2 0.85]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

test lr_scheduler_lambda-4.3 {Consistency - named syntax with different parameter aliases} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_lambda -optimizer $optimizer1 -multiplier 0.7]
    set scheduler2 [torch::lr_scheduler_lambda -optimizer $optimizer2 -lambda 0.7]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2]}
} {1}

test lr_scheduler_lambda-4.4 {Consistency - default multiplier behavior} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set optimizer3 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_lambda $optimizer1]
    set scheduler2 [torch::lr_scheduler_lambda -optimizer $optimizer2]
    set scheduler3 [torch::lrSchedulerLambda $optimizer3]
    
    expr {[is_valid_scheduler_handle $scheduler1] && [is_valid_scheduler_handle $scheduler2] && [is_valid_scheduler_handle $scheduler3]}
} {1}

# ===== Lambda Scheduler Functionality Tests =====

test lr_scheduler_lambda-5.1 {Default multiplier functionality} {
    set optimizer [create_test_optimizer]
    # Default multiplier should be 1.0
    set scheduler [torch::lr_scheduler_lambda $optimizer]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-5.2 {Decay multiplier (< 1.0)} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda $optimizer 0.95]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-5.3 {Growth multiplier (> 1.0)} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda $optimizer 1.05]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-5.4 {Unity multiplier (exactly 1.0)} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda $optimizer 1.0]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-5.5 {Very small multiplier} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda $optimizer 0.1]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-5.6 {Large multiplier} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda $optimizer 5.0]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-5.7 {Zero multiplier} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda $optimizer 0.0]
    
    is_valid_scheduler_handle $scheduler
} {1}

test lr_scheduler_lambda-5.8 {Negative multiplier} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_lambda $optimizer -0.5]
    
    is_valid_scheduler_handle $scheduler
} {1}

# ===== Error Handling Tests =====

test lr_scheduler_lambda-6.1 {Error handling - invalid optimizer} {
    set result [catch {
        torch::lr_scheduler_lambda "invalid_optimizer"
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_lambda-6.2 {Error handling - invalid optimizer with multiplier} {
    set result [catch {
        torch::lr_scheduler_lambda "invalid_optimizer" 0.9
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_lambda-6.3 {Error handling - invalid multiplier value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_lambda $optimizer "invalid_multiplier"
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_lambda-6.4 {Error handling - too many positional arguments} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_lambda $optimizer 0.9 extra_arg
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_lambda-6.5 {Error handling - named syntax missing parameter value} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_lambda -optimizer $optimizer -multiplier
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_lambda-6.6 {Error handling - unknown parameter} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_lambda -optimizer $optimizer -unknown_param 0.9
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_lambda-6.7 {Error handling - missing optimizer in named syntax} {
    set result [catch {
        torch::lr_scheduler_lambda -multiplier 0.9
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_lambda-6.8 {Error handling - invalid multiplier in named syntax} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_lambda -optimizer $optimizer -multiplier "invalid"
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

test lr_scheduler_lambda-6.9 {Error handling - odd number of parameters in named syntax} {
    set optimizer [create_test_optimizer]
    set result [catch {
        torch::lr_scheduler_lambda -optimizer $optimizer -multiplier 0.9 -extra
    } error]
    
    # Should return error
    expr {$result == 1}
} {1}

cleanupTests 