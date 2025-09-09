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

# Helper function to create test optimizer
proc create_test_optimizer {} {
    set weights [torch::tensor_create {1.0 2.0 3.0} float32]
    return [torch::optimizer_sgd $weights 0.1]
}

# Test cases for positional syntax
test lr_scheduler_onecycle_advanced-1.1 {Basic positional syntax - required params only} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-1.2 {Positional syntax with pct_start} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.05 2000 0.2]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-1.3 {Positional syntax with 5 parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.01 1500 0.4 "linear"]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-1.4 {Positional syntax with 6 parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.02 3000 0.25 "cos" 20.0]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-1.5 {Positional syntax with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.02 3000 0.25 "cos" 20.0 10.0]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-1.6 {Multiple schedulers with different max_lr} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_onecycle_advanced $optimizer1 0.1 1000]
    set scheduler2 [torch::lr_scheduler_onecycle_advanced $optimizer2 0.01 2000]
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

test lr_scheduler_onecycle_advanced-1.7 {Different total_steps values} {
    set optimizer [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 500]
    set scheduler2 [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 5000]
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

test lr_scheduler_onecycle_advanced-1.8 {Different pct_start values} {
    set optimizer [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000 0.1]
    set scheduler2 [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000 0.5]
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

test lr_scheduler_onecycle_advanced-1.9 {Different anneal_strategy values} {
    set optimizer [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000 0.3 "cos"]
    set scheduler2 [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000 0.3 "linear"]
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

# Test cases for camelCase alias
test lr_scheduler_onecycle_advanced-2.1 {camelCase alias - basic usage} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerOnecycleAdvanced $optimizer 0.1 1000]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-2.2 {camelCase alias with all parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerOnecycleAdvanced $optimizer 0.05 2000 0.25 "linear" 30.0 15.0]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-2.3 {camelCase alias consistency} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_onecycle_advanced $optimizer1 0.1 1000]
    set scheduler2 [torch::lrSchedulerOnecycleAdvanced $optimizer2 0.1 1000]
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

# Test cases for functionality
test lr_scheduler_onecycle_advanced-3.1 {Functional test with step} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000]
    # Step the scheduler multiple times
    torch::lr_scheduler_step_update $scheduler
    torch::lr_scheduler_step_update $scheduler
    torch::lr_scheduler_step_update $scheduler
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-3.2 {Functional test with different cycle lengths} {
    set optimizer [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 100]
    set scheduler2 [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 10000]
    torch::lr_scheduler_step_update $scheduler1
    torch::lr_scheduler_step_update $scheduler2
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

test lr_scheduler_onecycle_advanced-3.3 {Functional test with different strategies} {
    set optimizer [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000 0.3 "cos"]
    set scheduler2 [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000 0.3 "linear"]
    torch::lr_scheduler_step_update $scheduler1
    torch::lr_scheduler_step_update $scheduler2
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

test lr_scheduler_onecycle_advanced-3.4 {Functional test with multiple steps} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 500]
    for {set i 0} {$i < 10} {incr i} {
        torch::lr_scheduler_step_update $scheduler
    }
    expr {$scheduler ne ""}
} {1}

# Test cases for edge cases and validation
test lr_scheduler_onecycle_advanced-4.1 {Edge case - very small max_lr} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 1e-6 1000]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-4.2 {Edge case - large max_lr} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 10.0 1000]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-4.3 {Edge case - small total_steps} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 10]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-4.4 {Edge case - large total_steps} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 100000]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-4.5 {Edge case - minimum pct_start} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000 0.01]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-4.6 {Edge case - maximum pct_start} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000 0.99]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-4.7 {Edge case - with div_factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000 0.3 "cos" 2.0]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_onecycle_advanced-4.8 {Edge case - with final_div_factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000 0.3 "cos" 25.0 100.0]
    expr {$scheduler ne ""}
} {1}

# Test cases for error handling
test lr_scheduler_onecycle_advanced-5.1 {Error - missing arguments} {
    catch {torch::lr_scheduler_onecycle_advanced} result
    expr {[string match "*wrong # args*" $result]}
} {1}

test lr_scheduler_onecycle_advanced-5.2 {Error - missing max_lr} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_onecycle_advanced $optimizer} result
    expr {[string match "*wrong # args*" $result]}
} {1}

test lr_scheduler_onecycle_advanced-5.3 {Error - missing total_steps} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_onecycle_advanced $optimizer 0.1} result
    expr {[string match "*wrong # args*" $result]}
} {1}

test lr_scheduler_onecycle_advanced-5.4 {Error - invalid optimizer} {
    catch {torch::lr_scheduler_onecycle_advanced "invalid_handle" 0.1 1000} result
    expr {[string match "*Invalid optimizer*" $result]}
} {1}

test lr_scheduler_onecycle_advanced-5.5 {Error - invalid max_lr (non-numeric)} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_onecycle_advanced $optimizer "invalid" 1000} result
    expr {[string match "*expected floating-point*" $result]}
} {1}

test lr_scheduler_onecycle_advanced-5.6 {Error - invalid total_steps (non-numeric)} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_onecycle_advanced $optimizer 0.1 "invalid"} result
    expr {[string match "*expected integer*" $result]}
} {1}

test lr_scheduler_onecycle_advanced-5.7 {Error - invalid pct_start (non-numeric)} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000 "invalid"} result
    expr {[string match "*expected floating-point*" $result]}
} {1}

test lr_scheduler_onecycle_advanced-5.8 {Error - too many arguments} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000 0.3 "cos" 25.0 10.0 "extra"} result
    expr {[string match "*wrong # args*" $result]}
} {1}

test lr_scheduler_onecycle_advanced-5.9 {Error - not enough arguments} {
    catch {torch::lr_scheduler_onecycle_advanced optimizer_handle 0.1} result
    expr {[string match "*wrong # args*" $result]}
} {1}

cleanupTests 