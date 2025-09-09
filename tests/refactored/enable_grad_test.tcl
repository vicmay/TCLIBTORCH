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

# ============================================================================
# TORCH::ENABLE_GRAD COMMAND TESTS
# ============================================================================

# Test basic functionality - snake_case
test enable_grad-1.1 {Basic enable_grad functionality} {
    set result [torch::enable_grad]
    expr {$result eq "ok"}
} {1}

test enable_grad-1.2 {Enable grad returns ok string} {
    torch::enable_grad
} {ok}

test enable_grad-1.3 {Multiple enable_grad calls} {
    torch::enable_grad
    torch::enable_grad
    torch::enable_grad
} {ok}

# Test camelCase alias
test enable_grad-2.1 {CamelCase alias functionality} {
    set result [torch::enableGrad]
    expr {$result eq "ok"}
} {1}

test enable_grad-2.2 {CamelCase returns ok string} {
    torch::enableGrad
} {ok}

test enable_grad-2.3 {Multiple camelCase calls} {
    torch::enableGrad
    torch::enableGrad
    torch::enableGrad
} {ok}

# Test consistency between syntaxes
test enable_grad-3.1 {Snake_case and camelCase produce same result} {
    set result1 [torch::enable_grad]
    set result2 [torch::enableGrad]
    expr {$result1 eq $result2}
} {1}

test enable_grad-3.2 {Interleaved calls work correctly} {
    torch::enable_grad
    torch::enableGrad
    torch::enable_grad
    torch::enableGrad
} {ok}

# Test gradient state verification
test enable_grad-4.1 {Gradient enabled state after enable_grad} {
    torch::enable_grad
    torch::is_grad_enabled
} {1}

test enable_grad-4.2 {Gradient enabled state after camelCase} {
    torch::enableGrad
    torch::is_grad_enabled
} {1}

test enable_grad-4.3 {Enable after disable works} {
    torch::no_grad
    set disabled [torch::is_grad_enabled]
    torch::enable_grad
    set enabled [torch::is_grad_enabled]
    list $disabled $enabled
} {0 1}

test enable_grad-4.4 {CamelCase enable after disable works} {
    torch::no_grad
    set disabled [torch::is_grad_enabled]
    torch::enableGrad
    set enabled [torch::is_grad_enabled]
    list $disabled $enabled
} {0 1}

# Test error handling - wrong number of arguments
test enable_grad-5.1 {Error with extra arguments - snake_case} {
    catch {torch::enable_grad extra_arg} msg
    string match "*wrong # args*" $msg
} {1}

test enable_grad-5.2 {Error with extra arguments - camelCase} {
    catch {torch::enableGrad extra_arg} msg
    string match "*wrong # args*" $msg
} {1}

test enable_grad-5.3 {Error with multiple extra arguments - snake_case} {
    catch {torch::enable_grad arg1 arg2} msg
    string match "*wrong # args*" $msg
} {1}

test enable_grad-5.4 {Error with multiple extra arguments - camelCase} {
    catch {torch::enableGrad arg1 arg2} msg
    string match "*wrong # args*" $msg
} {1}

# Test integration with tensor operations
test enable_grad-6.1 {Enable grad affects tensor operations} {
    torch::enable_grad
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32 -requiresGrad true]
    set t2 [torch::tensorCreate -data {4.0 5.0 6.0} -shape {3} -dtype float32 -requiresGrad true]
    set result [torch::tensorAdd $t1 $t2]
    torch::tensorRequiresGrad $result
} {1}

test enable_grad-6.2 {CamelCase enable grad affects tensor operations} {
    torch::enableGrad
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32 -requiresGrad true]
    set t2 [torch::tensorCreate -data {4.0 5.0 6.0} -shape {3} -dtype float32 -requiresGrad true]
    set result [torch::tensorAdd $t1 $t2]
    torch::tensorRequiresGrad $result
} {1}

# Test with no_grad interaction
test enable_grad-7.1 {Enable after no_grad context} {
    torch::no_grad
    set grad_disabled [torch::is_grad_enabled]
    torch::enable_grad
    set grad_enabled [torch::is_grad_enabled]
    list $grad_disabled $grad_enabled
} {0 1}

test enable_grad-7.2 {CamelCase enable after no_grad context} {
    torch::no_grad
    set grad_disabled [torch::is_grad_enabled]
    torch::enableGrad
    set grad_enabled [torch::is_grad_enabled]
    list $grad_disabled $grad_enabled
} {0 1}

# Test idempotency
test enable_grad-8.1 {Multiple enable calls are idempotent} {
    torch::enable_grad
    set state1 [torch::is_grad_enabled]
    torch::enable_grad
    set state2 [torch::is_grad_enabled]
    torch::enable_grad
    set state3 [torch::is_grad_enabled]
    list $state1 $state2 $state3
} {1 1 1}

test enable_grad-8.2 {Multiple camelCase enable calls are idempotent} {
    torch::enableGrad
    set state1 [torch::is_grad_enabled]
    torch::enableGrad
    set state2 [torch::is_grad_enabled]
    torch::enableGrad
    set state3 [torch::is_grad_enabled]
    list $state1 $state2 $state3
} {1 1 1}

# Test mixed usage patterns
test enable_grad-9.1 {Mixed snake_case and camelCase usage} {
    torch::no_grad
    torch::enable_grad
    set state1 [torch::is_grad_enabled]
    torch::no_grad
    torch::enableGrad
    set state2 [torch::is_grad_enabled]
    list $state1 $state2
} {1 1}

test enable_grad-9.2 {Complex enable/disable sequence} {
    # Start with enabled
    torch::enable_grad
    set s1 [torch::is_grad_enabled]
    
    # Disable with no_grad
    torch::no_grad
    set s2 [torch::is_grad_enabled]
    
    # Re-enable with camelCase
    torch::enableGrad
    set s3 [torch::is_grad_enabled]
    
    # Disable again
    torch::no_grad
    set s4 [torch::is_grad_enabled]
    
    # Final enable with snake_case
    torch::enable_grad
    set s5 [torch::is_grad_enabled]
    
    list $s1 $s2 $s3 $s4 $s5
} {1 0 1 0 1}

# Test command existence
test enable_grad-10.1 {Snake_case command exists} {
    info commands torch::enable_grad
} {::torch::enable_grad}

test enable_grad-10.2 {CamelCase command exists} {
    info commands torch::enableGrad
} {::torch::enableGrad}

test enable_grad-10.3 {Both commands are different objects} {
    expr {[info commands torch::enable_grad] ne [info commands torch::enableGrad]}
} {1}

cleanupTests 