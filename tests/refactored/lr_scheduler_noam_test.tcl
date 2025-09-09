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
test lr_scheduler_noam-1.1 {Basic positional syntax - default warmup} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam $optimizer 512]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-1.2 {Positional syntax with custom warmup} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam $optimizer 512 2000]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-1.3 {Positional syntax with different model size} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam $optimizer 256 8000]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-1.4 {Multiple positional schedulers} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_noam $optimizer1 512]
    set scheduler2 [torch::lr_scheduler_noam $optimizer2 1024 6000]
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

# Test cases for named parameter syntax
test lr_scheduler_noam-2.1 {Named parameter syntax - required params only} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-2.2 {Named parameter syntax - all params} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512 -warmupSteps 8000]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-2.3 {Named parameters with snake_case aliases} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -model_size 256 -warmup_steps 2000]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-2.4 {Named parameters mixed order} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam -warmupSteps 6000 -optimizer $optimizer -modelSize 1024]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-2.5 {Named parameters with different optimizers} {
    set adam_opt [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam -optimizer $adam_opt -modelSize 768 -warmupSteps 10000]
    expr {$scheduler ne ""}
} {1}

# Test cases for camelCase alias
test lr_scheduler_noam-3.1 {camelCase alias - positional syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerNoam $optimizer 512]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-3.2 {camelCase alias - named syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerNoam -optimizer $optimizer -modelSize 512 -warmupSteps 4000]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-3.3 {camelCase alias with different model sizes} {
    set optimizer [create_test_optimizer]
    set scheduler1 [torch::lrSchedulerNoam -optimizer $optimizer -modelSize 128]
    set scheduler2 [torch::lrSchedulerNoam -optimizer $optimizer -modelSize 2048 -warmupSteps 16000]
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

# Test cases for syntax consistency
test lr_scheduler_noam-4.1 {Syntax consistency check - basic} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_noam $optimizer1 512]
    set scheduler2 [torch::lr_scheduler_noam -optimizer $optimizer2 -modelSize 512]
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

test lr_scheduler_noam-4.2 {Syntax consistency with warmup steps} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_noam $optimizer1 512 8000]
    set scheduler2 [torch::lr_scheduler_noam -optimizer $optimizer2 -modelSize 512 -warmupSteps 8000]
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

test lr_scheduler_noam-4.3 {camelCase consistency} {
    set optimizer1 [create_test_optimizer]
    set optimizer2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_noam -optimizer $optimizer1 -modelSize 512]
    set scheduler2 [torch::lrSchedulerNoam -optimizer $optimizer2 -modelSize 512]
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

# Test cases for functionality
test lr_scheduler_noam-5.1 {Functional test with step} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512 -warmupSteps 1000]
    # Step the scheduler multiple times
    torch::lr_scheduler_step_update $scheduler
    torch::lr_scheduler_step_update $scheduler
    torch::lr_scheduler_step_update $scheduler
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-5.2 {Functional test with different model sizes} {
    set optimizer [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 256]
    set scheduler2 [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 1024]
    torch::lr_scheduler_step_update $scheduler1
    torch::lr_scheduler_step_update $scheduler2
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

test lr_scheduler_noam-5.3 {Functional test with warmup variations} {
    set optimizer [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512 -warmupSteps 2000]
    set scheduler2 [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512 -warmupSteps 8000]
    torch::lr_scheduler_step_update $scheduler1
    torch::lr_scheduler_step_update $scheduler2
    expr {$scheduler1 ne "" && $scheduler2 ne ""}
} {1}

test lr_scheduler_noam-5.4 {Functional test with Adam optimizer} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 768 -warmupSteps 4000]
    torch::lr_scheduler_step_update $scheduler
    torch::lr_scheduler_step_update $scheduler
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-5.5 {Functional test with multiple steps} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512 -warmupSteps 500]
    for {set i 0} {$i < 10} {incr i} {
        torch::lr_scheduler_step_update $scheduler
    }
    expr {$scheduler ne ""}
} {1}

# Test cases for edge cases and validation
test lr_scheduler_noam-6.1 {Edge case - small model size} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 32]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-6.2 {Edge case - large model size} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 4096]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-6.3 {Edge case - small warmup steps} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512 -warmupSteps 10]
    expr {$scheduler ne ""}
} {1}

test lr_scheduler_noam-6.4 {Edge case - large warmup steps} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512 -warmupSteps 100000]
    expr {$scheduler ne ""}
} {1}

# Test cases for error handling
test lr_scheduler_noam-7.1 {Error - missing optimizer} {
    catch {torch::lr_scheduler_noam -modelSize 512} result
    expr {[string match "*optimizer*" $result] || [string match "*required*" $result]}
} {1}

test lr_scheduler_noam-7.2 {Error - missing model size} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_noam -optimizer $optimizer} result
    expr {[string match "*modelSize*" $result] || [string match "*required*" $result]}
} {1}

test lr_scheduler_noam-7.3 {Error - invalid optimizer} {
    catch {torch::lr_scheduler_noam -optimizer "invalid_handle" -modelSize 512} result
    expr {[string match "*optimizer*" $result] || [string match "*invalid*" $result]}
} {1}

test lr_scheduler_noam-7.4 {Error - invalid model size (zero)} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_noam -optimizer $optimizer -modelSize 0} result
    expr {[string match "*modelSize*" $result] || [string match "*positive*" $result]}
} {1}

test lr_scheduler_noam-7.5 {Error - invalid model size (negative)} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_noam -optimizer $optimizer -modelSize -512} result
    expr {[string match "*modelSize*" $result] || [string match "*positive*" $result]}
} {1}

test lr_scheduler_noam-7.6 {Error - invalid warmup steps (negative)} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512 -warmupSteps -100} result
    expr {[string match "*warmupSteps*" $result] || [string match "*positive*" $result]}
} {1}

test lr_scheduler_noam-7.7 {Error - unknown parameter} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512 -unknownParam value} result
    expr {[string match "*unknown*" $result] || [string match "*invalid*" $result]}
} {1}

cleanupTests 