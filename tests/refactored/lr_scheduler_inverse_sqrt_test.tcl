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

test lr_scheduler_inverse_sqrt-1.1 {Basic positional syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 4000]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-1.2 {Positional with warmup steps} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 8000]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-1.3 {Positional with warmup and decay factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 2000 0.5]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-1.4 {Positional with different decay factors} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 1000 2.0]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

# 2. Named parameter syntax tests

test lr_scheduler_inverse_sqrt-2.1 {Named parameter syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt -optimizer $optimizer -warmupSteps 4000]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-2.2 {Named parameters with decay factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt -optimizer $optimizer -warmupSteps 8000 -decayFactor 0.8]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-2.3 {Named parameters in different order} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt -decayFactor 1.5 -warmupSteps 2000 -optimizer $optimizer]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-2.4 {Named syntax with only required parameter} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt -optimizer $optimizer -warmupSteps 6000]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-2.5 {Alternative parameter names (snake_case)} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt -optimizer $optimizer -warmup_steps 4000 -decay_factor 0.9]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

# 3. camelCase alias tests

test lr_scheduler_inverse_sqrt-3.1 {camelCase alias with positional syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerInverseSqrt $optimizer 4000]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-3.2 {camelCase alias with named parameters} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerInverseSqrt -optimizer $optimizer -warmupSteps 5000 -decayFactor 0.7]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-3.3 {camelCase alias parameter order variation} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerInverseSqrt -warmupSteps 3000 -decayFactor 1.2 -optimizer $optimizer]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

# 4. Error handling tests

test lr_scheduler_inverse_sqrt-4.1 {Error on invalid optimizer handle} {
    catch {torch::lr_scheduler_inverse_sqrt invalid_handle 4000} result
    string match "*Invalid optimizer name*" $result
} {1}

test lr_scheduler_inverse_sqrt-4.2 {Error on missing warmup_steps in positional} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_inverse_sqrt $optimizer} result
    string match "*Usage: torch::lr_scheduler_inverse_sqrt optimizer warmup_steps*" $result
} {1}

test lr_scheduler_inverse_sqrt-4.3 {Error on negative warmup_steps} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_inverse_sqrt $optimizer -1000} result
    string match "*Required parameters missing or invalid*" $result
} {1}

test lr_scheduler_inverse_sqrt-4.4 {Error on zero warmup_steps} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_inverse_sqrt $optimizer 0} result
    string match "*Required parameters missing or invalid*" $result
} {1}

test lr_scheduler_inverse_sqrt-4.5 {Error on negative decay factor} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_inverse_sqrt $optimizer 4000 -0.5} result
    string match "*Required parameters missing or invalid*" $result
} {1}

test lr_scheduler_inverse_sqrt-4.6 {Error on zero decay factor} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_inverse_sqrt $optimizer 4000 0.0} result
    string match "*Required parameters missing or invalid*" $result
} {1}

test lr_scheduler_inverse_sqrt-4.7 {Error on missing optimizer in named syntax} {
    catch {torch::lr_scheduler_inverse_sqrt -warmupSteps 4000} result
    string match "*Required parameters missing or invalid*" $result
} {1}

test lr_scheduler_inverse_sqrt-4.8 {Error on missing warmup_steps in named syntax} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_inverse_sqrt -optimizer $optimizer} result
    string match "*Required parameters missing or invalid*" $result
} {1}

test lr_scheduler_inverse_sqrt-4.9 {Error on unknown parameter} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_inverse_sqrt -optimizer $optimizer -warmupSteps 4000 -unknown_param value} result
    string match "*Unknown parameter*" $result
} {1}

test lr_scheduler_inverse_sqrt-4.10 {Error on invalid warmup_steps string} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_inverse_sqrt $optimizer "invalid"} result
    string match "*Invalid warmup_steps value*" $result
} {1}

test lr_scheduler_inverse_sqrt-4.11 {Error on invalid decay_factor string} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_inverse_sqrt $optimizer 4000 "invalid"} result
    string match "*Invalid decay_factor value*" $result
} {1}

test lr_scheduler_inverse_sqrt-4.12 {Error on unpaired named parameters} {
    set optimizer [create_test_optimizer]
    catch {torch::lr_scheduler_inverse_sqrt -optimizer $optimizer -warmupSteps} result
    string match "*Named parameters must be in pairs*" $result
} {1}

# 5. Warmup steps parameter tests

test lr_scheduler_inverse_sqrt-5.1 {Small warmup steps} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 100]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-5.2 {Large warmup steps} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 50000]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-5.3 {Common warmup steps values} {
    set optimizer [create_test_optimizer]
    foreach warmup {1000 2000 4000 8000 16000} {
        set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer $warmup]
        if {![string match "inverse_sqrt_scheduler*" $scheduler]} {
            error "Failed for warmup $warmup"
        }
    }
    set result "pass"
} {pass}

# 6. Decay factor parameter tests

test lr_scheduler_inverse_sqrt-6.1 {Very small decay factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 4000 0.01]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-6.2 {Large decay factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 4000 10.0]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-6.3 {Common decay factor values} {
    set optimizer [create_test_optimizer]
    foreach decay {0.1 0.5 1.0 1.5 2.0 5.0} {
        set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 4000 $decay]
        if {![string match "inverse_sqrt_scheduler*" $scheduler]} {
            error "Failed for decay factor $decay"
        }
    }
    set result "pass"
} {pass}

# 7. Multiple optimizer support tests

test lr_scheduler_inverse_sqrt-7.1 {Multiple optimizers with different parameters} {
    set opt1 [create_test_optimizer]
    set opt2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_inverse_sqrt $opt1 4000 1.0]
    set scheduler2 [torch::lr_scheduler_inverse_sqrt $opt2 8000 0.5]
    expr {[string match "inverse_sqrt_scheduler*" $scheduler1] && [string match "inverse_sqrt_scheduler*" $scheduler2]}
} {1}

test lr_scheduler_inverse_sqrt-7.2 {Multiple schedulers for same optimizer} {
    set optimizer [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_inverse_sqrt $optimizer 4000 1.0]
    set scheduler2 [torch::lr_scheduler_inverse_sqrt $optimizer 8000 2.0]
    expr {[string match "inverse_sqrt_scheduler*" $scheduler1] && [string match "inverse_sqrt_scheduler*" $scheduler2]}
} {1}

# 8. Syntax consistency tests

test lr_scheduler_inverse_sqrt-8.1 {Consistency between positional and named syntax} {
    set opt1 [create_test_optimizer]
    set opt2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_inverse_sqrt $opt1 4000 0.8]
    set scheduler2 [torch::lr_scheduler_inverse_sqrt -optimizer $opt2 -warmupSteps 4000 -decayFactor 0.8]
    # Both should return valid scheduler handles
    expr {[string match "inverse_sqrt_scheduler*" $scheduler1] && [string match "inverse_sqrt_scheduler*" $scheduler2]}
} {1}

test lr_scheduler_inverse_sqrt-8.2 {Consistency between snake_case and camelCase} {
    set opt1 [create_test_optimizer]
    set opt2 [create_test_optimizer]
    set scheduler1 [torch::lr_scheduler_inverse_sqrt $opt1 4000 1.5]
    set scheduler2 [torch::lrSchedulerInverseSqrt $opt2 4000 1.5]
    # Both should return valid scheduler handles
    expr {[string match "inverse_sqrt_scheduler*" $scheduler1] && [string match "inverse_sqrt_scheduler*" $scheduler2]}
} {1}

# 9. Edge case tests

test lr_scheduler_inverse_sqrt-9.1 {Very precise decay factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 4000 0.987654321]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-9.2 {Scientific notation decay factor} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 4000 1.5e0]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-9.3 {Maximum reasonable warmup steps} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 1000000]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

# 10. Parameter validation tests

test lr_scheduler_inverse_sqrt-10.1 {Invalid optimizer handle with named syntax} {
    catch {torch::lr_scheduler_inverse_sqrt -optimizer "invalid_opt" -warmupSteps 4000} result
    string match "*Invalid optimizer name*" $result
} {1}

test lr_scheduler_inverse_sqrt-10.2 {Empty optimizer handle} {
    catch {torch::lr_scheduler_inverse_sqrt -optimizer "" -warmupSteps 4000} result
    string match "*Required parameters missing or invalid*" $result
} {1}

# 11. Documentation examples verification

test lr_scheduler_inverse_sqrt-11.1 {README example - positional syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 4000]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-11.2 {README example - named syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt -optimizer $optimizer -warmupSteps 4000 -decayFactor 1.0]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-11.3 {README example - camelCase} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lrSchedulerInverseSqrt -optimizer $optimizer -warmupSteps 4000 -decayFactor 1.0]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

# 12. Default parameter tests

test lr_scheduler_inverse_sqrt-12.1 {Default decay factor with named syntax} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt -optimizer $optimizer -warmupSteps 4000]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

test lr_scheduler_inverse_sqrt-12.2 {Explicit default values} {
    set optimizer [create_test_optimizer]
    set scheduler [torch::lr_scheduler_inverse_sqrt -optimizer $optimizer -warmupSteps 4000 -decayFactor 1.0]
    string match "inverse_sqrt_scheduler*" $scheduler
} {1}

cleanupTests 