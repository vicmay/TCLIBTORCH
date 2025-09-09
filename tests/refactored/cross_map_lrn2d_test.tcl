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
# Test torch::cross_map_lrn2d - Dual Syntax Support
# ============================================================================

# Test setup - create test tensors
# NCHW format for 2D convolution
set test_2d [torch::randn {1 3 4 4}]
# Batch of 2
set test_3d [torch::randn {2 3 4 4}]
# Minimal tensor
set test_small [torch::randn {1 1 2 2}]

# ============================================================================
# Test Positional Syntax (Backward Compatibility)
# ============================================================================

test cross_map_lrn2d-1.1 {Basic positional syntax with 2D tensor} {
    set result [torch::cross_map_lrn2d $test_2d 5 1e-4 0.75 1.0]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-1.2 {Positional syntax with different parameters} {
    set result [torch::cross_map_lrn2d $test_2d 3 0.001 0.5 2.0]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-1.3 {Positional syntax with batch tensor} {
    set result [torch::cross_map_lrn2d $test_3d 5 1e-4 0.75 1.0]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-1.4 {Positional syntax error - wrong number of args} {
    catch {torch::cross_map_lrn2d $test_2d 5 1e-4} error
    string match "*Wrong number of arguments*" $error
} 1

test cross_map_lrn2d-1.5 {Positional syntax error - too many args} {
    catch {torch::cross_map_lrn2d $test_2d 5 1e-4 0.75 1.0 extra} error
    string match "*Wrong number of arguments*" $error
} 1

test cross_map_lrn2d-1.6 {Positional syntax error - invalid tensor} {
    catch {torch::cross_map_lrn2d invalid_tensor 5 1e-4 0.75 1.0} error
    string match "*Tensor not found*" $error
} 1

test cross_map_lrn2d-1.7 {Positional syntax error - invalid size} {
    catch {torch::cross_map_lrn2d $test_2d invalid_size 1e-4 0.75 1.0} error
    string match "*Invalid size parameter*" $error
} 1

test cross_map_lrn2d-1.8 {Positional syntax error - invalid alpha} {
    catch {torch::cross_map_lrn2d $test_2d 5 invalid_alpha 0.75 1.0} error
    string match "*Invalid alpha parameter*" $error
} 1

# ============================================================================
# Test Named Parameter Syntax
# ============================================================================

test cross_map_lrn2d-2.1 {Named parameter syntax with all parameters} {
    set result [torch::cross_map_lrn2d -input $test_2d -size 5 -alpha 1e-4 -beta 0.75 -k 1.0]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-2.2 {Named parameter syntax with defaults} {
    set result [torch::cross_map_lrn2d -input $test_2d]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-2.3 {Named parameter syntax with partial parameters} {
    set result [torch::cross_map_lrn2d -input $test_2d -size 3 -alpha 0.001]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-2.4 {Named parameter syntax with different order} {
    set result [torch::cross_map_lrn2d -beta 0.5 -input $test_2d -alpha 0.001 -size 7 -k 2.0]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-2.5 {Named parameter syntax error - missing input} {
    catch {torch::cross_map_lrn2d -size 5 -alpha 1e-4} error
    string match "*Required parameter missing*" $error
} 1

test cross_map_lrn2d-2.6 {Named parameter syntax error - invalid parameter} {
    catch {torch::cross_map_lrn2d -input $test_2d -invalid_param value} error
    string match "*Unknown parameter*" $error
} 1

test cross_map_lrn2d-2.7 {Named parameter syntax error - missing value} {
    catch {torch::cross_map_lrn2d -input $test_2d -size} error
    string match "*Missing value for parameter*" $error
} 1

test cross_map_lrn2d-2.8 {Named parameter syntax error - invalid tensor} {
    catch {torch::cross_map_lrn2d -input invalid_tensor} error
    string match "*Tensor not found*" $error
} 1

# ============================================================================
# Test camelCase Alias
# ============================================================================

test cross_map_lrn2d-3.1 {camelCase alias with positional syntax} {
    set result [torch::crossMapLrn2d $test_2d 5 1e-4 0.75 1.0]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-3.2 {camelCase alias with named syntax} {
    set result [torch::crossMapLrn2d -input $test_2d -size 5 -alpha 1e-4 -beta 0.75 -k 1.0]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-3.3 {camelCase alias with defaults} {
    set result [torch::crossMapLrn2d -input $test_2d]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-3.4 {camelCase alias error handling} {
    catch {torch::crossMapLrn2d invalid_tensor 5 1e-4 0.75 1.0} error
    string match "*Tensor not found*" $error
} 1

# ============================================================================
# Test Syntax Consistency (Both syntaxes produce same results)
# ============================================================================

test cross_map_lrn2d-4.1 {Syntax consistency - same parameters} {
    set pos_result [torch::cross_map_lrn2d $test_2d 5 1e-4 0.75 1.0]
    set named_result [torch::cross_map_lrn2d -input $test_2d -size 5 -alpha 1e-4 -beta 0.75 -k 1.0]
    set camel_result [torch::crossMapLrn2d -input $test_2d -size 5 -alpha 1e-4 -beta 0.75 -k 1.0]
    
    # All should return valid tensor handles
    expr {[string match "tensor*" $pos_result] && [string match "tensor*" $named_result] && [string match "tensor*" $camel_result]}
} 1

test cross_map_lrn2d-4.2 {Syntax consistency - with defaults} {
    set named_result [torch::cross_map_lrn2d -input $test_2d]
    set pos_result [torch::cross_map_lrn2d $test_2d 5 1e-4 0.75 1.0]
    
    # Both should return valid tensor handles
    expr {[string match "tensor*" $named_result] && [string match "tensor*" $pos_result]}
} 1

# ============================================================================
# Test Different Parameter Values
# ============================================================================

test cross_map_lrn2d-5.1 {Different size values} {
    set result1 [torch::cross_map_lrn2d -input $test_2d -size 3]
    set result2 [torch::cross_map_lrn2d -input $test_2d -size 7]
    set result3 [torch::cross_map_lrn2d -input $test_2d -size 1]
    
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2] && [string match "tensor*" $result3]}
} 1

test cross_map_lrn2d-5.2 {Different alpha values} {
    set result1 [torch::cross_map_lrn2d -input $test_2d -alpha 1e-3]
    set result2 [torch::cross_map_lrn2d -input $test_2d -alpha 1e-5]
    set result3 [torch::cross_map_lrn2d -input $test_2d -alpha 1e-1]
    
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2] && [string match "tensor*" $result3]}
} 1

test cross_map_lrn2d-5.3 {Different beta values} {
    set result1 [torch::cross_map_lrn2d -input $test_2d -beta 0.5]
    set result2 [torch::cross_map_lrn2d -input $test_2d -beta 1.0]
    set result3 [torch::cross_map_lrn2d -input $test_2d -beta 0.25]
    
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2] && [string match "tensor*" $result3]}
} 1

test cross_map_lrn2d-5.4 {Different k values} {
    set result1 [torch::cross_map_lrn2d -input $test_2d -k 0.5]
    set result2 [torch::cross_map_lrn2d -input $test_2d -k 2.0]
    set result3 [torch::cross_map_lrn2d -input $test_2d -k 5.0]
    
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2] && [string match "tensor*" $result3]}
} 1

# ============================================================================
# Test Edge Cases
# ============================================================================

test cross_map_lrn2d-6.1 {Edge case - minimum size} {
    set result [torch::cross_map_lrn2d -input $test_small -size 1]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-6.2 {Edge case - zero alpha} {
    set result [torch::cross_map_lrn2d -input $test_2d -alpha 0.0]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-6.3 {Edge case - large parameters} {
    set result [torch::cross_map_lrn2d -input $test_2d -size 11 -alpha 0.1 -beta 2.0 -k 10.0]
    string match "tensor*" $result
} 1

test cross_map_lrn2d-6.4 {Edge case - invalid size (zero)} {
    catch {torch::cross_map_lrn2d -input $test_2d -size 0} error
    string match "*Required parameter missing*" $error
} 1

test cross_map_lrn2d-6.5 {Edge case - negative size} {
    catch {torch::cross_map_lrn2d -input $test_2d -size -1} error
    string match "*Required parameter missing*" $error
} 1

# ============================================================================
# Test Output Shape and Properties
# ============================================================================

test cross_map_lrn2d-7.1 {Output shape preservation} {
    set input_shape [torch::tensor_shape $test_2d]
    set result [torch::cross_map_lrn2d -input $test_2d]
    set output_shape [torch::tensor_shape $result]
    
    expr {$input_shape == $output_shape}
} 1

test cross_map_lrn2d-7.2 {Output dtype preservation} {
    set input_dtype [torch::tensor_dtype $test_2d]
    set result [torch::cross_map_lrn2d -input $test_2d]
    set output_dtype [torch::tensor_dtype $result]
    
    expr {$input_dtype == $output_dtype}
} 1

test cross_map_lrn2d-7.3 {Output device preservation} {
    set input_device [torch::tensor_device $test_2d]
    set result [torch::cross_map_lrn2d -input $test_2d]
    set output_device [torch::tensor_device $result]
    
    # Output device should match input device
    expr {$input_device == $output_device}
} 1

# ============================================================================
# Performance Test (Basic)
# ============================================================================

test cross_map_lrn2d-8.1 {Performance test - multiple calls} {
    set start_time [clock milliseconds]
    for {set i 0} {$i < 100} {incr i} {
        torch::cross_map_lrn2d $test_2d 5 1e-4 0.75 1.0
    }
    set end_time [clock milliseconds]
    set duration [expr {$end_time - $start_time}]
    expr {$duration < 5000}
} 1

# Cleanup
cleanupTests 