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
# Tests for torch::lr_scheduler_multiplicative - Multiplicative Learning Rate Scheduler
# ============================================================================

# Since we don't have real optimizers for testing, we'll focus on parameter
# parsing and error handling. Most tests will expect "Invalid optimizer name"
# error, but the important thing is that the parsing works correctly.

# ============================================================================
# Tests for Positional Syntax (Backward Compatibility)
# ============================================================================

test lr_scheduler_multiplicative-1.1 {Positional syntax - parameter parsing works} {
    # Test that positional syntax is parsed correctly (invalid optimizer is expected)
    catch {torch::lr_scheduler_multiplicative test_opt 0.95} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-1.2 {Positional syntax - different lr_lambda values} {
    catch {torch::lr_scheduler_multiplicative test_opt 0.8} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-1.3 {Positional syntax - wrong number of args} {
    catch {torch::lr_scheduler_multiplicative test_opt} result
    string match "*Usage: torch::lr_scheduler_multiplicative*" $result
} {1}

test lr_scheduler_multiplicative-1.4 {Positional syntax - too many args} {
    catch {torch::lr_scheduler_multiplicative test_opt 0.5 extra} result
    string match "*Usage: torch::lr_scheduler_multiplicative*" $result
} {1}

test lr_scheduler_multiplicative-1.5 {Positional syntax - negative lr_lambda} {
    catch {torch::lr_scheduler_multiplicative test_opt -0.1} result
    set result
} {Invalid optimizer name}

# ============================================================================
# Tests for Named Parameter Syntax
# ============================================================================

test lr_scheduler_multiplicative-2.1 {Named parameter syntax - basic parsing} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 0.9} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-2.2 {Named parameter syntax - with lr_lambda parameter} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lr_lambda 0.85} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-2.3 {Named parameter syntax - different order} {
    catch {torch::lr_scheduler_multiplicative -lrLambda 0.75 -optimizer test_opt} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-2.4 {Named parameter syntax - missing optimizer} {
    catch {torch::lr_scheduler_multiplicative -lrLambda 0.9} result
    string match "*Required parameters missing*" $result
} {1}

test lr_scheduler_multiplicative-2.5 {Named parameter syntax - only optimizer (should use default)} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt} result
    set result
} {Invalid optimizer name}

# ============================================================================
# Tests for camelCase Alias
# ============================================================================

test lr_scheduler_multiplicative-3.1 {camelCase alias - positional syntax} {
    catch {torch::lrSchedulerMultiplicative test_opt 0.9} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-3.2 {camelCase alias - named parameters} {
    catch {torch::lrSchedulerMultiplicative -optimizer test_opt -lrLambda 0.8} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-3.3 {camelCase alias exists} {
    # Just check that the command exists
    expr {[llength [info commands torch::lrSchedulerMultiplicative]] > 0}
} {1}

# ============================================================================
# Tests for Error Handling
# ============================================================================

test lr_scheduler_multiplicative-4.1 {Error handling - unknown parameter} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 0.9 -invalid param} result
    string match "*Unknown parameter: -invalid*" $result
} {1}

test lr_scheduler_multiplicative-4.2 {Error handling - invalid lr_lambda value} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda invalid} result
    string match "*Invalid lr_lambda value*" $result
} {1}

test lr_scheduler_multiplicative-4.3 {Error handling - missing value for parameter} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda} result
    string match "*Missing value for parameter*" $result
} {1}

test lr_scheduler_multiplicative-4.4 {Error handling - invalid lr_lambda in positional} {
    catch {torch::lr_scheduler_multiplicative test_opt invalid_value} result
    string match "*Invalid lr_lambda value*" $result
} {1}

# ============================================================================
# Tests for Parameter Validation
# ============================================================================

test lr_scheduler_multiplicative-5.1 {Parameter validation - positive lr_lambda} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 0.95} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-5.2 {Parameter validation - lr_lambda = 1.0} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 1.0} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-5.3 {Parameter validation - small lr_lambda} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 0.01} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-5.4 {Parameter validation - large lr_lambda} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 2.0} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-5.5 {Parameter validation - zero lr_lambda} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 0.0} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-5.6 {Parameter validation - negative lr_lambda} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda -0.5} result
    set result
} {Invalid optimizer name}

# ============================================================================
# Tests for Different Parameter Names
# ============================================================================

test lr_scheduler_multiplicative-6.1 {Different parameter names - lrLambda} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 0.9} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-6.2 {Different parameter names - lr_lambda} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lr_lambda 0.9} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-6.3 {Parameter equivalence - both forms should work} {
    # Test that both parameter names are accepted
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 0.5} result1
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lr_lambda 0.5} result2
    
    # Both should give the same result (invalid optimizer error)
    expr {$result1 eq $result2}
} {1}

# ============================================================================
# Tests for Syntax Consistency
# ============================================================================

test lr_scheduler_multiplicative-7.1 {Consistency check - both syntaxes give same result type} {
    catch {torch::lr_scheduler_multiplicative test_opt 0.9} result1
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 0.9} result2
    
    # Both should produce the same error
    expr {$result1 eq $result2}
} {1}

test lr_scheduler_multiplicative-7.2 {Consistency between snake_case and camelCase} {
    catch {torch::lr_scheduler_multiplicative test_opt 0.8} result1
    catch {torch::lrSchedulerMultiplicative test_opt 0.8} result2
    
    # Both should produce the same error
    expr {$result1 eq $result2}
} {1}

# ============================================================================
# Command Existence Tests
# ============================================================================

test lr_scheduler_multiplicative-8.1 {Command exists - snake_case} {
    expr {[llength [info commands torch::lr_scheduler_multiplicative]] > 0}
} {1}

test lr_scheduler_multiplicative-8.2 {Command exists - camelCase alias} {
    expr {[llength [info commands torch::lrSchedulerMultiplicative]] > 0}
} {1}

# ============================================================================
# Mathematical Edge Cases
# ============================================================================

test lr_scheduler_multiplicative-9.1 {Mathematical edge case - very small lr_lambda} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 1e-10} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-9.2 {Mathematical edge case - very large lr_lambda} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 1000.0} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multiplicative-9.3 {Mathematical edge case - fractional lr_lambda} {
    catch {torch::lr_scheduler_multiplicative -optimizer test_opt -lrLambda 0.33333} result
    set result
} {Invalid optimizer name}

cleanupTests 