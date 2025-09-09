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

# Helper function to create a test optimizer
proc create_test_optimizer {} {
    # Create some test parameters
    set param1 [torch::ones -shape {3 3} -dtype float32 -requires_grad 1]
    set param2 [torch::ones -shape {5} -dtype float32 -requires_grad 1]
    
    # Create SGD optimizer
    set opt [torch::optimizer_sgd [list $param1 $param2] 0.01]
    return $opt
}

# Helper function to create another test optimizer with different LR
proc create_test_optimizer_custom_lr {lr} {
    # Create some test parameters
    set param1 [torch::ones -shape {2 2} -dtype float32 -requires_grad 1]
    
    # Create Adam optimizer  
    set opt [torch::optimizer_adam [list $param1] $lr]
    return $opt
}

# Test 1: Positional syntax - basic functionality
test lr_scheduler_constant_with_warmup-1.1 {Basic positional syntax} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lr_scheduler_constant_with_warmup $opt 100]
    expr {[string length $scheduler] > 0}
} {1}

# Test 2: Positional syntax - with last_epoch
test lr_scheduler_constant_with_warmup-1.2 {Positional syntax with last_epoch} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lr_scheduler_constant_with_warmup $opt 50 10]
    expr {[string length $scheduler] > 0}
} {1}

# Test 3: Positional syntax - minimal parameters
test lr_scheduler_constant_with_warmup-1.3 {Positional syntax minimal parameters} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lr_scheduler_constant_with_warmup $opt 200]
    expr {[string length $scheduler] > 0}
} {1}

# Test 4: Named parameter syntax - basic functionality
test lr_scheduler_constant_with_warmup-2.1 {Named parameter syntax basic} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps 100]
    expr {[string length $scheduler] > 0}
} {1}

# Test 5: Named parameter syntax - all parameters
test lr_scheduler_constant_with_warmup-2.2 {Named parameter syntax all parameters} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lr_scheduler_constant_with_warmup \
        -optimizer $opt \
        -numWarmupSteps 50 \
        -lastEpoch 5]
    expr {[string length $scheduler] > 0}
} {1}

# Test 6: Named parameter syntax - snake_case aliases
test lr_scheduler_constant_with_warmup-2.3 {Named parameter syntax with snake_case aliases} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lr_scheduler_constant_with_warmup \
        -optimizer $opt \
        -num_warmup_steps 75 \
        -last_epoch 3]
    expr {[string length $scheduler] > 0}
} {1}

# Test 7: Named parameter syntax - mixed order
test lr_scheduler_constant_with_warmup-2.4 {Named parameter syntax mixed order} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lr_scheduler_constant_with_warmup \
        -lastEpoch 8 \
        -optimizer $opt \
        -numWarmupSteps 120]
    expr {[string length $scheduler] > 0}
} {1}

# Test 8: camelCase alias - basic functionality
test lr_scheduler_constant_with_warmup-3.1 {camelCase alias basic functionality} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lrSchedulerConstantWithWarmup $opt 100]
    expr {[string length $scheduler] > 0}
} {1}

# Test 9: camelCase alias - with named parameters
test lr_scheduler_constant_with_warmup-3.2 {camelCase alias with named parameters} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lrSchedulerConstantWithWarmup \
        -optimizer $opt \
        -numWarmupSteps 80 \
        -lastEpoch 2]
    expr {[string length $scheduler] > 0}
} {1}

# Test 10: camelCase alias - positional with last_epoch
test lr_scheduler_constant_with_warmup-3.3 {camelCase alias positional with last_epoch} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lrSchedulerConstantWithWarmup $opt 60 15]
    expr {[string length $scheduler] > 0}
} {1}

# Test 11: Both syntaxes should work with same parameters
test lr_scheduler_constant_with_warmup-4.1 {Both syntaxes produce valid schedulers} {
    set opt1 [create_test_optimizer]
    set opt2 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_constant_with_warmup $opt1 100 5]
    set scheduler2 [torch::lr_scheduler_constant_with_warmup \
        -optimizer $opt2 \
        -numWarmupSteps 100 \
        -lastEpoch 5]
    
    # Both should produce valid scheduler handles
    expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0}
} {1}

# Test 12: Different warmup steps
test lr_scheduler_constant_with_warmup-4.2 {Different warmup steps} {
    set opt [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps 10]
    set scheduler2 [torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps 1000]
    
    # Both should succeed
    expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0}
} {1}

# Test 13: Different last_epoch values
test lr_scheduler_constant_with_warmup-4.3 {Different last_epoch values} {
    set opt1 [create_test_optimizer]
    set opt2 [create_test_optimizer]
    set opt3 [create_test_optimizer]
    
    set scheduler1 [torch::lr_scheduler_constant_with_warmup -optimizer $opt1 -numWarmupSteps 50 -lastEpoch -1]
    set scheduler2 [torch::lr_scheduler_constant_with_warmup -optimizer $opt2 -numWarmupSteps 50 -lastEpoch 0]
    set scheduler3 [torch::lr_scheduler_constant_with_warmup -optimizer $opt3 -numWarmupSteps 50 -lastEpoch 100]
    
    # All should succeed
    expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0 && [string length $scheduler3] > 0}
} {1}

# Test 14: Integration with different optimizers
test lr_scheduler_constant_with_warmup-4.4 {Integration with different optimizers} {
    set sgd_opt [create_test_optimizer]
    set adam_opt [create_test_optimizer_custom_lr 0.001]
    
    set scheduler1 [torch::lr_scheduler_constant_with_warmup $sgd_opt 50]
    set scheduler2 [torch::lr_scheduler_constant_with_warmup $adam_opt 100]
    
    # Both should work
    expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0}
} {1}

# Test 15: Zero warmup steps (edge case)
test lr_scheduler_constant_with_warmup-4.5 {Zero warmup steps edge case} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps 0]
    expr {[string length $scheduler] > 0}
} {1}

# Test 16: Large warmup steps
test lr_scheduler_constant_with_warmup-4.6 {Large warmup steps} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps 100000]
    expr {[string length $scheduler] > 0}
} {1}

# Test 17: Error handling - missing required parameters (positional)
test lr_scheduler_constant_with_warmup-5.1 {Error handling - too few positional arguments} {
    set opt [create_test_optimizer]
    
    set caught 0
    if {[catch {torch::lr_scheduler_constant_with_warmup $opt} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 18: Error handling - missing required parameters (named)
test lr_scheduler_constant_with_warmup-5.2 {Error handling - missing optimizer parameter} {
    set caught 0
    if {[catch {torch::lr_scheduler_constant_with_warmup -numWarmupSteps 100} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 19: Error handling - missing warmup steps parameter
test lr_scheduler_constant_with_warmup-5.3 {Error handling - missing warmup steps parameter} {
    set opt [create_test_optimizer]
    
    set caught 0
    if {[catch {torch::lr_scheduler_constant_with_warmup -optimizer $opt} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 20: Error handling - invalid optimizer name
test lr_scheduler_constant_with_warmup-5.4 {Error handling - invalid optimizer name} {
    set caught 0
    if {[catch {torch::lr_scheduler_constant_with_warmup -optimizer invalid_opt -numWarmupSteps 100} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 21: Error handling - invalid warmup steps type
test lr_scheduler_constant_with_warmup-5.5 {Error handling - invalid warmup steps type} {
    set opt [create_test_optimizer]
    
    set caught 0
    if {[catch {torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps invalid} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 22: Error handling - negative warmup steps
test lr_scheduler_constant_with_warmup-5.6 {Error handling - negative warmup steps} {
    set opt [create_test_optimizer]
    
    set caught 0
    if {[catch {torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps -10} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 23: Error handling - invalid last_epoch type
test lr_scheduler_constant_with_warmup-5.7 {Error handling - invalid last_epoch type} {
    set opt [create_test_optimizer]
    
    set caught 0
    if {[catch {torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps 100 -lastEpoch invalid} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 24: Error handling - unknown parameter
test lr_scheduler_constant_with_warmup-5.8 {Error handling - unknown parameter} {
    set opt [create_test_optimizer]
    
    set caught 0
    if {[catch {torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps 100 -unknown_param value} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 25: Error handling - too many positional arguments
test lr_scheduler_constant_with_warmup-5.9 {Error handling - too many positional arguments} {
    set opt [create_test_optimizer]
    
    set caught 0
    if {[catch {torch::lr_scheduler_constant_with_warmup $opt 100 5 extra_arg} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 26: Error handling - missing parameter value
test lr_scheduler_constant_with_warmup-5.10 {Error handling - missing parameter value} {
    set opt [create_test_optimizer]
    
    set caught 0
    if {[catch {torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps} result]} {
        set caught 1
    }
    expr {$caught == 1}
} {1}

# Test 27: Data type compatibility - different optimizer types
test lr_scheduler_constant_with_warmup-6.1 {Different optimizer data types} {
    # Create optimizer with different learning rates
    set opt1 [create_test_optimizer_custom_lr 0.1]
    set opt2 [create_test_optimizer_custom_lr 0.001]
    set opt3 [create_test_optimizer_custom_lr 1.0]
    
    set scheduler1 [torch::lr_scheduler_constant_with_warmup $opt1 50]
    set scheduler2 [torch::lr_scheduler_constant_with_warmup $opt2 50]
    set scheduler3 [torch::lr_scheduler_constant_with_warmup $opt3 50]
    
    # All should work regardless of initial LR
    expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0 && [string length $scheduler3] > 0}
} {1}

# Test 28: Parameter consistency verification
test lr_scheduler_constant_with_warmup-6.2 {Parameter consistency between syntaxes} {
    set opt1 [create_test_optimizer]
    set opt2 [create_test_optimizer]
    
    # Test same parameters in both syntaxes
    set scheduler1 [torch::lr_scheduler_constant_with_warmup $opt1 100 10]
    set scheduler2 [torch::lr_scheduler_constant_with_warmup \
        -optimizer $opt2 \
        -numWarmupSteps 100 \
        -lastEpoch 10]
    
    # Both should succeed with equivalent parameters
    expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0}
} {1}

# Test 29: Edge case - maximum reasonable warmup steps
test lr_scheduler_constant_with_warmup-6.3 {Edge case - large warmup steps} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps 1000000]
    expr {[string length $scheduler] > 0}
} {1}

# Test 30: Edge case - negative last_epoch
test lr_scheduler_constant_with_warmup-6.4 {Edge case - negative last_epoch} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps 50 -lastEpoch -100]
    expr {[string length $scheduler] > 0}
} {1}

# Test 31: Multiple schedulers for same optimizer
test lr_scheduler_constant_with_warmup-6.5 {Multiple schedulers for same optimizer} {
    set opt [create_test_optimizer]
    
    # Create multiple schedulers for the same optimizer (should be allowed)
    set scheduler1 [torch::lr_scheduler_constant_with_warmup $opt 50]
    set scheduler2 [torch::lr_scheduler_constant_with_warmup $opt 100]
    set scheduler3 [torch::lr_scheduler_constant_with_warmup $opt 200]
    
    # All should succeed
    expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0 && [string length $scheduler3] > 0}
} {1}

# Test 32: Scheduler handle format verification
test lr_scheduler_constant_with_warmup-6.6 {Scheduler handle format verification} {
    set opt [create_test_optimizer]
    
    set scheduler [torch::lr_scheduler_constant_with_warmup $opt 100]
    
    # Check that handle contains expected prefix
    set contains_prefix [string match "*constant_warmup_scheduler*" $scheduler]
    expr {$contains_prefix == 1}
} {1}

# Test 33: Both parameter name styles work
test lr_scheduler_constant_with_warmup-6.7 {Both parameter name styles work} {
    set opt1 [create_test_optimizer]
    set opt2 [create_test_optimizer]
    
    # Test camelCase style parameters
    set scheduler1 [torch::lr_scheduler_constant_with_warmup \
        -optimizer $opt1 \
        -numWarmupSteps 50 \
        -lastEpoch 5]
    
    # Test snake_case style parameters
    set scheduler2 [torch::lr_scheduler_constant_with_warmup \
        -optimizer $opt2 \
        -num_warmup_steps 50 \
        -last_epoch 5]
    
    # Both should work
    expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0}
} {1}

# Test 34: Boundary values
test lr_scheduler_constant_with_warmup-6.8 {Boundary values testing} {
    set opt1 [create_test_optimizer]
    set opt2 [create_test_optimizer]
    set opt3 [create_test_optimizer]
    
    # Test boundary values
    set scheduler1 [torch::lr_scheduler_constant_with_warmup $opt1 1]
    set scheduler2 [torch::lr_scheduler_constant_with_warmup $opt2 0]
    set scheduler3 [torch::lr_scheduler_constant_with_warmup $opt3 50 0]
    
    # All should work
    expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0 && [string length $scheduler3] > 0}
} {1}

# Test 35: Complex integration test
test lr_scheduler_constant_with_warmup-7.1 {Complex integration test} {
    # Create multiple optimizers with different configurations
    set sgd_opt [create_test_optimizer]
    set adam_opt [create_test_optimizer_custom_lr 0.001]
    
    # Create schedulers with different syntaxes
    set scheduler1 [torch::lr_scheduler_constant_with_warmup $sgd_opt 100 10]
    set scheduler2 [torch::lrSchedulerConstantWithWarmup \
        -optimizer $adam_opt \
        -numWarmupSteps 200 \
        -lastEpoch 5]
    
    # Both should work
    expr {[string length $scheduler1] > 0 && [string length $scheduler2] > 0}
} {1}

# Test 36: Backward compatibility verification
test lr_scheduler_constant_with_warmup-7.2 {Backward compatibility verification} {
    set opt [create_test_optimizer]
    
    # Old positional syntax should still work exactly as before
    set scheduler [torch::lr_scheduler_constant_with_warmup $opt 100]
    expr {[string length $scheduler] > 0}
} {1}

cleanupTests 