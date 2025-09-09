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
# Test torch::count_parameters - Dual Syntax Support
# ============================================================================

# Test setup - create test models using global variables
set linear_model [torch::linear -inFeatures 10 -outFeatures 5 -bias true]

# Create individual layers for sequential model
set layer1 [torch::linear -inFeatures 10 -outFeatures 20 -bias true]
set layer2 [torch::linear -inFeatures 20 -outFeatures 1 -bias true]
set seq_model [torch::sequential [list $layer1 $layer2]]

# ============================================================================
# Test Positional Syntax (Backward Compatibility)
# ============================================================================

test count_parameters-1.1 {Basic positional syntax with linear model} {
    set param_count [torch::count_parameters $linear_model]
    # Linear layer: (10 * 5) + 5 = 55 parameters (weights + bias)
    expr {$param_count == 55}
} 1

test count_parameters-1.2 {Positional syntax with sequential model} {
    set param_count [torch::count_parameters $seq_model]
    # Layer 1: (10 * 20) + 20 = 220
    # Layer 2: (20 * 1) + 1 = 21
    # Total: 220 + 21 = 241
    expr {$param_count == 241}
} 1

test count_parameters-1.3 {Positional syntax error - wrong number of args} {
    catch {torch::count_parameters} error
    string match "*Required parameter missing*" $error
} 1

test count_parameters-1.4 {Positional syntax error - too many args} {
    catch {torch::count_parameters $linear_model extra_arg} error
    string match "*Wrong number of arguments*" $error
} 1

test count_parameters-1.5 {Positional syntax error - invalid model} {
    catch {torch::count_parameters invalid_model} error
    string match "*Model not found*" $error
} 1

# ============================================================================
# Test Named Parameter Syntax
# ============================================================================

test count_parameters-2.1 {Named parameter syntax with linear model} {
    set param_count [torch::count_parameters -model $linear_model]
    expr {$param_count == 55}
} 1

test count_parameters-2.2 {Named parameter syntax with sequential model} {
    set param_count [torch::count_parameters -model $seq_model]
    expr {$param_count == 241}
} 1

test count_parameters-2.3 {Named parameter syntax error - missing model} {
    catch {torch::count_parameters -model} error
    string match "*Missing value for parameter*" $error
} 1

test count_parameters-2.4 {Named parameter syntax error - invalid parameter} {
    catch {torch::count_parameters -invalid_param $linear_model} error
    string match "*Unknown parameter*" $error
} 1

test count_parameters-2.5 {Named parameter syntax error - missing required parameter} {
    catch {torch::count_parameters -other value} error
    string match "*Unknown parameter*" $error
} 1

test count_parameters-2.6 {Named parameter syntax error - invalid model} {
    catch {torch::count_parameters -model invalid_model} error
    string match "*Model not found*" $error
} 1

# ============================================================================
# Test camelCase Alias
# ============================================================================

test count_parameters-3.1 {camelCase alias with positional syntax} {
    set param_count [torch::countParameters $linear_model]
    expr {$param_count == 55}
} 1

test count_parameters-3.2 {camelCase alias with named syntax} {
    set param_count [torch::countParameters -model $linear_model]
    expr {$param_count == 55}
} 1

test count_parameters-3.3 {camelCase alias with sequential model} {
    set param_count [torch::countParameters -model $seq_model]
    expr {$param_count == 241}
} 1

test count_parameters-3.4 {camelCase alias error handling} {
    catch {torch::countParameters invalid_model} error
    string match "*Model not found*" $error
} 1

# ============================================================================
# Test Syntax Consistency (Both syntaxes produce same results)
# ============================================================================

test count_parameters-4.1 {Syntax consistency - linear model} {
    set pos_result [torch::count_parameters $linear_model]
    set named_result [torch::count_parameters -model $linear_model]
    set camel_result [torch::countParameters -model $linear_model]
    expr {$pos_result == $named_result && $named_result == $camel_result}
} 1

test count_parameters-4.2 {Syntax consistency - sequential model} {
    set pos_result [torch::count_parameters $seq_model]
    set named_result [torch::count_parameters -model $seq_model]
    set camel_result [torch::countParameters $seq_model]
    expr {$pos_result == $named_result && $named_result == $camel_result}
} 1

# ============================================================================
# Test Edge Cases
# ============================================================================

test count_parameters-5.1 {Model with no parameters - check empty model} {
    # Create an empty sequential model
    set empty_model [torch::sequential]
    set param_count [torch::count_parameters $empty_model]
    expr {$param_count == 0}
} 1

test count_parameters-5.2 {Model with bias disabled} {
    set no_bias_model [torch::linear -inFeatures 5 -outFeatures 3 -bias false]
    set param_count [torch::count_parameters $no_bias_model]
    # Only weights: 5 * 3 = 15 parameters
    expr {$param_count == 15}
} 1

test count_parameters-5.3 {Large parameter count} {
    set large_model [torch::linear -inFeatures 1000 -outFeatures 1000 -bias true]
    set param_count [torch::count_parameters $large_model]
    # (1000 * 1000) + 1000 = 1,001,000 parameters
    expr {$param_count == 1001000}
} 1

# ============================================================================
# Test Return Value Types
# ============================================================================

test count_parameters-6.1 {Return value is integer} {
    set param_count [torch::count_parameters $linear_model]
    string is integer $param_count
} 1

test count_parameters-6.2 {Return value is non-negative} {
    set param_count [torch::count_parameters $linear_model]
    expr {$param_count >= 0}
} 1

# ============================================================================
# Performance Test (Basic)
# ============================================================================

test count_parameters-7.1 {Performance test - multiple calls} {
    set start_time [clock milliseconds]
    for {set i 0} {$i < 1000} {incr i} {
        torch::count_parameters $linear_model
    }
    set end_time [clock milliseconds]
    set duration [expr {$end_time - $start_time}]
    # Should complete 1000 calls in reasonable time (< 5 seconds)
    expr {$duration < 5000}
} 1

# Cleanup
cleanupTests 