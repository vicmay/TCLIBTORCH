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

# Helper to create a simple optimizer for testing
proc create_test_optimizer {} {
    set param1 [torch::zeros {3 3}]
    set param2 [torch::zeros {5}]
    set opt [torch::optimizer_sgd [list $param1 $param2] 0.01]
    return $opt
}

# 1. Positional syntax tests (backward compatibility)

test lr_scheduler_exponential_decay-1.1 {Basic positional syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.9]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-1.2 {Default gamma with positional syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.95]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-1.3 {Small gamma value} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.1]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-1.4 {Large gamma value} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.999]
    string match "scheduler*" $scheduler
} {1}

# 2. Named parameter syntax tests

test lr_scheduler_exponential_decay-2.1 {Named parameter syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay -optimizer $optimizer -gamma 0.9]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-2.2 {Named parameters in different order} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay -gamma 0.8 -optimizer $optimizer]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-2.3 {Named syntax with default gamma omitted} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay -optimizer $optimizer]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-2.4 {Named syntax with explicit default gamma} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay -optimizer $optimizer -gamma 0.95]
    string match "scheduler*" $scheduler
} {1}

# 3. camelCase alias tests

test lr_scheduler_exponential_decay-3.1 {camelCase alias with positional syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerExponentialDecay $optimizer 0.85]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-3.2 {camelCase alias with named parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerExponentialDecay -optimizer $optimizer -gamma 0.75]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-3.3 {camelCase alias parameter order variation} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerExponentialDecay -gamma 0.6 -optimizer $optimizer]
    string match "scheduler*" $scheduler
} {1}

# 4. Error handling tests

test lr_scheduler_exponential_decay-4.1 {Error on invalid optimizer handle} {
    catch {torch::lr_scheduler_exponential_decay invalid_handle 0.9} result
    string match "*Invalid optimizer handle*" $result
} {1}

test lr_scheduler_exponential_decay-4.2 {Error on negative gamma} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_exponential_decay $optimizer -0.1} result
    string match "*Required parameters missing or invalid*" $result
} {1}

test lr_scheduler_exponential_decay-4.3 {Error on gamma greater than 1} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_exponential_decay $optimizer 1.5} result
    string match "*Required parameters missing or invalid*" $result
} {1}

test lr_scheduler_exponential_decay-4.4 {Error on zero gamma} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_exponential_decay $optimizer 0.0} result
    string match "*Required parameters missing or invalid*" $result
} {1}

test lr_scheduler_exponential_decay-4.5 {Error on missing optimizer in named syntax} {
    catch {torch::lr_scheduler_exponential_decay -gamma 0.9} result
    string match "*Required parameters missing or invalid*" $result
} {1}

test lr_scheduler_exponential_decay-4.6 {Error on unknown parameter} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_exponential_decay -optimizer $optimizer -unknown_param value} result
    string match "*Unknown parameter*" $result
} {1}

test lr_scheduler_exponential_decay-4.7 {Error on invalid gamma string} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_exponential_decay $optimizer "invalid"} result
    string match "*Invalid gamma value*" $result
} {1}

test lr_scheduler_exponential_decay-4.8 {Error on unpaired named parameters} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_exponential_decay -optimizer $optimizer -gamma} result
    string match "*Named parameters must be in pairs*" $result
} {1}

# 5. Gamma boundary value tests

test lr_scheduler_exponential_decay-5.1 {Minimum valid gamma} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.001]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-5.2 {Maximum valid gamma} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay $optimizer 1.0]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-5.3 {Common gamma values} {
    set optimizer [create_test_optimizer]
    foreach gamma {0.1 0.2 0.5 0.7 0.9 0.95 0.99} {
        set scheduler [torch::lr_scheduler_exponential_decay $optimizer $gamma]
        if {![string match "scheduler*" $scheduler]} {
            error "Failed for gamma $gamma"
        }
    }
    set result "pass"
} {pass}

# 6. Multiple optimizer support tests

test lr_scheduler_exponential_decay-6.1 {Multiple optimizers with different gammas} {
    set opt1 [create_test_optimizer]
    set opt2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_exponential_decay $opt1 0.9]
    set scheduler2 [torch::lr_scheduler_exponential_decay $opt2 0.8]
    expr {[string match "scheduler*" $scheduler1] && [string match "scheduler*" $scheduler2]}
} {1}

test lr_scheduler_exponential_decay-6.2 {Multiple schedulers for same optimizer} {
    set optimizer [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_exponential_decay $optimizer 0.9]
    set scheduler2 [torch::lr_scheduler_exponential_decay $optimizer 0.8]
    expr {[string match "scheduler*" $scheduler1] && [string match "scheduler*" $scheduler2]}
} {1}

# 7. Syntax consistency tests

test lr_scheduler_exponential_decay-7.1 {Consistency between positional and named syntax} {
    set opt1 [create_test_optimizer]
    set opt2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_exponential_decay $opt1 0.85]
    set scheduler2 [torch::lr_scheduler_exponential_decay -optimizer $opt2 -gamma 0.85]
    # Both should return valid scheduler handles
    expr {[string match "scheduler*" $scheduler1] && [string match "scheduler*" $scheduler2]}
} {1}

test lr_scheduler_exponential_decay-7.2 {Consistency between snake_case and camelCase} {
    set opt1 [create_test_optimizer]
    set opt2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_exponential_decay $opt1 0.7]
    set scheduler2 [torch::lrSchedulerExponentialDecay $opt2 0.7]
    # Both should return valid scheduler handles
    expr {[string match "scheduler*" $scheduler1] && [string match "scheduler*" $scheduler2]}
} {1}

# 8. Edge case tests

test lr_scheduler_exponential_decay-8.1 {Very precise gamma value} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.987654321]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-8.2 {Scientific notation gamma} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay $optimizer 9.5e-1]
    string match "scheduler*" $scheduler
} {1}

# 9. Parameter validation tests

test lr_scheduler_exponential_decay-9.1 {Invalid optimizer handle with named syntax} {
    catch {torch::lr_scheduler_exponential_decay -optimizer "invalid_opt" -gamma 0.9} result
    string match "*Invalid optimizer handle*" $result
} {1}

test lr_scheduler_exponential_decay-9.2 {Empty optimizer handle} {
    catch {torch::lr_scheduler_exponential_decay -optimizer "" -gamma 0.9} result
    string match "*Required parameters missing or invalid*" $result
} {1}

# 10. Documentation examples verification

test lr_scheduler_exponential_decay-10.1 {README example - positional syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.95]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-10.2 {README example - named syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_exponential_decay -optimizer $optimizer -gamma 0.95]
    string match "scheduler*" $scheduler
} {1}

test lr_scheduler_exponential_decay-10.3 {README example - camelCase} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerExponentialDecay -optimizer $optimizer -gamma 0.95]
    string match "scheduler*" $scheduler
} {1}

cleanupTests 