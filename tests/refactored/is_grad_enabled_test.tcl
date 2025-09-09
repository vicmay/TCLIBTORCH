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
# TORCH::IS_GRAD_ENABLED COMMAND TESTS
# ============================================================================

# Test basic functionality - snake_case
test is_grad_enabled-1.1 {Basic is_grad_enabled functionality} {
    torch::enable_grad
    torch::is_grad_enabled
} {1}

test is_grad_enabled-1.2 {is_grad_enabled returns boolean} {
    torch::enable_grad
    set result [torch::is_grad_enabled]
    expr {$result == 0 || $result == 1}
} {1}

test is_grad_enabled-1.3 {is_grad_enabled with disabled gradients} {
    torch::no_grad
    torch::is_grad_enabled
} {0}

# Test camelCase alias
test is_grad_enabled-2.1 {CamelCase alias functionality - enabled} {
    torch::enable_grad
    torch::isGradEnabled
} {1}

test is_grad_enabled-2.2 {CamelCase alias functionality - disabled} {
    torch::no_grad
    torch::isGradEnabled
} {0}

test is_grad_enabled-2.3 {CamelCase returns boolean} {
    torch::enable_grad
    set result [torch::isGradEnabled]
    expr {$result == 0 || $result == 1}
} {1}

# Test consistency between syntaxes
test is_grad_enabled-3.1 {Snake_case and camelCase produce same result - enabled} {
    torch::enable_grad
    set result1 [torch::is_grad_enabled]
    set result2 [torch::isGradEnabled]
    expr {$result1 == $result2}
} {1}

test is_grad_enabled-3.2 {Snake_case and camelCase produce same result - disabled} {
    torch::no_grad
    set result1 [torch::is_grad_enabled]
    set result2 [torch::isGradEnabled]
    expr {$result1 == $result2}
} {1}

test is_grad_enabled-3.3 {Multiple calls produce consistent results} {
    torch::enable_grad
    set r1 [torch::is_grad_enabled]
    set r2 [torch::isGradEnabled]
    set r3 [torch::is_grad_enabled]
    set r4 [torch::isGradEnabled]
    expr {$r1 == $r2 && $r2 == $r3 && $r3 == $r4}
} {1}

# Test error handling - wrong number of arguments
test is_grad_enabled-4.1 {Error with extra arguments - snake_case} {
    catch {torch::is_grad_enabled extra_arg} msg
    string match "*wrong # args*" $msg
} {1}

test is_grad_enabled-4.2 {Error with extra arguments - camelCase} {
    catch {torch::isGradEnabled extra_arg} msg
    string match "*wrong # args*" $msg
} {1}

test is_grad_enabled-4.3 {Error with multiple extra arguments - snake_case} {
    catch {torch::is_grad_enabled arg1 arg2} msg
    string match "*wrong # args*" $msg
} {1}

test is_grad_enabled-4.4 {Error with multiple extra arguments - camelCase} {
    catch {torch::isGradEnabled arg1 arg2} msg
    string match "*wrong # args*" $msg
} {1}

# Test state detection with different gradient contexts
test is_grad_enabled-5.1 {Detects enabled state after enable_grad} {
    torch::enable_grad
    torch::is_grad_enabled
} {1}

test is_grad_enabled-5.2 {Detects disabled state after no_grad} {
    torch::no_grad
    torch::is_grad_enabled
} {0}

test is_grad_enabled-5.3 {Detects state changes} {
    torch::enable_grad
    set enabled [torch::is_grad_enabled]
    torch::no_grad
    set disabled [torch::is_grad_enabled]
    list $enabled $disabled
} {1 0}

test is_grad_enabled-5.4 {CamelCase detects state changes} {
    torch::enable_grad
    set enabled [torch::isGradEnabled]
    torch::no_grad
    set disabled [torch::isGradEnabled]
    list $enabled $disabled
} {1 0}

# Test with set_grad_enabled
test is_grad_enabled-6.1 {Detects state set by set_grad_enabled true} {
    torch::set_grad_enabled true
    torch::is_grad_enabled
} {1}

test is_grad_enabled-6.2 {Detects state set by set_grad_enabled false} {
    torch::set_grad_enabled false
    torch::is_grad_enabled
} {0}

test is_grad_enabled-6.3 {CamelCase detects set_grad_enabled changes} {
    torch::set_grad_enabled true
    set enabled [torch::isGradEnabled]
    torch::set_grad_enabled false
    set disabled [torch::isGradEnabled]
    list $enabled $disabled
} {1 0}

# Test integration with tensor operations
test is_grad_enabled-7.1 {Gradient state affects tensor creation} {
    torch::enable_grad
    set is_enabled [torch::is_grad_enabled]
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32 -requiresGrad true]
    set requires_grad [torch::tensorRequiresGrad $t1]
    list $is_enabled $requires_grad
} {1 1}

test is_grad_enabled-7.2 {Gradient disabled state with tensors} {
    torch::no_grad
    set is_enabled [torch::is_grad_enabled]
    ;# Note: requiresGrad can still be set even when gradients are globally disabled
    set t1 [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32 -requiresGrad true]
    set requires_grad [torch::tensorRequiresGrad $t1]
    list $is_enabled $requires_grad
} {0 1}

# Test complex state transitions
test is_grad_enabled-8.1 {Complex enable/disable sequence tracking} {
    # Start disabled
    torch::no_grad
    set s1 [torch::is_grad_enabled]
    
    # Enable
    torch::enable_grad
    set s2 [torch::is_grad_enabled]
    
    # Disable with set_grad_enabled
    torch::set_grad_enabled false
    set s3 [torch::is_grad_enabled]
    
    # Enable with camelCase
    torch::enableGrad
    set s4 [torch::is_grad_enabled]
    
    # Check with camelCase
    set s5 [torch::isGradEnabled]
    
    list $s1 $s2 $s3 $s4 $s5
} {0 1 0 1 1}

test is_grad_enabled-8.2 {Idempotent operations don't change result} {
    torch::enable_grad
    set s1 [torch::is_grad_enabled]
    torch::enable_grad
    set s2 [torch::is_grad_enabled]
    torch::enable_grad
    set s3 [torch::is_grad_enabled]
    list $s1 $s2 $s3
} {1 1 1}

test is_grad_enabled-8.3 {Multiple disable operations} {
    torch::enable_grad
    torch::no_grad
    set s1 [torch::is_grad_enabled]
    torch::no_grad
    set s2 [torch::is_grad_enabled]
    torch::set_grad_enabled false
    set s3 [torch::is_grad_enabled]
    list $s1 $s2 $s3
} {0 0 0}

# Test mixed usage patterns
test is_grad_enabled-9.1 {Mixed snake_case and camelCase checking} {
    torch::enable_grad
    set snake_result [torch::is_grad_enabled]
    set camel_result [torch::isGradEnabled]
    torch::no_grad
    set snake_disabled [torch::is_grad_enabled]
    set camel_disabled [torch::isGradEnabled]
    
    expr {$snake_result == $camel_result && $snake_disabled == $camel_disabled && $snake_result == 1 && $snake_disabled == 0}
} {1}

# Test thread safety (basic check)
test is_grad_enabled-10.1 {Multiple rapid calls return consistent results} {
    torch::enable_grad
    set results {}
    for {set i 0} {$i < 10} {incr i} {
        lappend results [torch::is_grad_enabled]
    }
    # All results should be 1
    set all_ones 1
    foreach result $results {
        if {$result != 1} {
            set all_ones 0
            break
        }
    }
    set all_ones
} {1}

test is_grad_enabled-10.2 {Mixed rapid calls with camelCase} {
    torch::enable_grad
    set results {}
    for {set i 0} {$i < 5} {incr i} {
        lappend results [torch::is_grad_enabled]
        lappend results [torch::isGradEnabled]
    }
    # All results should be 1
    set all_ones 1
    foreach result $results {
        if {$result != 1} {
            set all_ones 0
            break
        }
    }
    set all_ones
} {1}

cleanupTests 