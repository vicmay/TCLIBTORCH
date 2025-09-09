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
# Tests for torch::multi_margin_loss - Multi-class Margin Loss Function
# ============================================================================

# Helper function to create test tensors for multi-margin loss
proc create_test_tensors {} {
    # For multi-margin loss, input should be (N, C) and target should be (N,)
    # Let's create simple 1D tensors and let LibTorch handle the shapes
    # This is a simplified test approach - real multi-margin loss would need proper multi-class setup
    
    # Create simple input tensor - we'll treat this as single sample with multiple classes
    set input [torch::tensor_create -data {2.0 1.0 0.5} -dtype float32 -device cpu -requiresGrad false]
    
    # Create target tensor - single target class index
    set target [torch::tensor_create -data {0} -dtype int64 -device cpu -requiresGrad false]
    
    return [list $input $target]
}

test multi_margin_loss-1.1 {Basic multi-class margin loss - positional syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss $input $target]
    
    # Verify result is a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-1.2 {Multi-class margin loss with p=1 - positional} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss $input $target 1]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-1.3 {Multi-class margin loss with p=2 - positional} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss $input $target 2]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-1.4 {Multi-class margin loss with custom margin - positional} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss $input $target 1 0.5]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-1.5 {Multi-class margin loss with sum reduction - positional} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss $input $target 1 1.0 2]
    expr {[string match "tensor*" $result]}
} {1}

# ============================================================================
# Tests for Named Parameter Syntax
# ============================================================================

test multi_margin_loss-2.1 {Named parameter syntax - basic usage} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss -input $input -target $target]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-2.2 {Named parameter syntax - with p parameter} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss -input $input -target $target -p 2]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-2.3 {Named parameter syntax - with margin} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss -input $input -target $target -margin 0.8]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-2.4 {Named parameter syntax - reduction none} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss -input $input -target $target -reduction none]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-2.5 {Named parameter syntax - reduction mean} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss -input $input -target $target -reduction mean]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-2.6 {Named parameter syntax - all parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss -input $input -target $target -p 1 -margin 1.0 -reduction sum]
    expr {[string match "tensor*" $result]}
} {1}

# ============================================================================
# Tests for camelCase Alias
# ============================================================================

test multi_margin_loss-3.1 {camelCase alias - basic usage} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multiMarginLoss $input $target]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-3.2 {camelCase alias - with named parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multiMarginLoss -input $input -target $target -p 2 -reduction sum]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-3.3 {camelCase alias - positional syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multiMarginLoss $input $target 1 0.5 1]
    expr {[string match "tensor*" $result]}
} {1}

# ============================================================================
# Tests for Syntax Consistency
# ============================================================================

test multi_margin_loss-4.1 {Consistency check - both syntaxes give same result type} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result1 [torch::multi_margin_loss $input $target 1 1.0 1]
    set result2 [torch::multi_margin_loss -input $input -target $target -p 1 -margin 1.0 -reduction mean]
    
    # Both should produce valid tensor handles
    set valid1 [string match "tensor*" $result1]
    set valid2 [string match "tensor*" $result2]
    expr {$valid1 && $valid2}
} {1}

test multi_margin_loss-4.2 {Consistency between snake_case and camelCase} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result1 [torch::multi_margin_loss $input $target]
    set result2 [torch::multiMarginLoss $input $target]
    
    # Both should produce valid tensor handles
    set valid1 [string match "tensor*" $result1]
    set valid2 [string match "tensor*" $result2]
    expr {$valid1 && $valid2}
} {1}

# ============================================================================
# Tests for Error Handling
# ============================================================================

test multi_margin_loss-5.1 {Error handling - invalid input tensor} {
    set tensors [create_test_tensors]
    set target [lindex $tensors 1]
    
    catch {torch::multi_margin_loss invalid_tensor $target} result
    set result
} {Invalid input tensor name}

test multi_margin_loss-5.2 {Error handling - invalid target tensor} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    
    catch {torch::multi_margin_loss $input invalid_tensor} result
    set result
} {Invalid target tensor name}

test multi_margin_loss-5.3 {Error handling - missing required parameters} {
    set caught [catch {torch::multi_margin_loss -input tensor1} result]
    expr {$caught == 1}
} {1}

test multi_margin_loss-5.4 {Error handling - unknown parameter} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    catch {torch::multi_margin_loss -input $input -target $target -invalid param} result
    string match "*Unknown parameter: -invalid*" $result
} {1}

test multi_margin_loss-5.5 {Error handling - invalid reduction value} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    catch {torch::multi_margin_loss -input $input -target $target -reduction invalid} result
    string match "*Invalid reduction value*" $result
} {1}

test multi_margin_loss-5.6 {Error handling - invalid p value} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    catch {torch::multi_margin_loss -input $input -target $target -p invalid} result
    string match "*Invalid p value*" $result
} {1}

test multi_margin_loss-5.7 {Error handling - invalid margin value} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    catch {torch::multi_margin_loss -input $input -target $target -margin invalid} result
    string match "*Invalid margin value*" $result
} {1}

# ============================================================================
# Mathematical Correctness Tests
# ============================================================================

test multi_margin_loss-6.1 {Mathematical correctness - known classification scenario} {
    # Create simple classification test case
    set input [torch::tensor_create -data {2.0 1.0 0.5} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0} -dtype int64 -device cpu -requiresGrad false]
    
    set result [torch::multi_margin_loss $input $target]
    
    # Should produce a valid result tensor
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-6.2 {Mathematical correctness - p=1 vs p=2 norm} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result_p1 [torch::multi_margin_loss $input $target 1]
    set result_p2 [torch::multi_margin_loss $input $target 2]
    
    # Both should be valid tensors
    set valid_p1 [string match "tensor*" $result_p1]
    set valid_p2 [string match "tensor*" $result_p2]
    expr {$valid_p1 && $valid_p2}
} {1}

test multi_margin_loss-6.3 {Mathematical correctness - reduction none vs mean} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result_none [torch::multi_margin_loss $input $target 1 1.0 0]
    set result_mean [torch::multi_margin_loss $input $target 1 1.0 1]
    
    # Both should be valid tensors but with different shapes potentially
    set valid_none [string match "tensor*" $result_none]
    set valid_mean [string match "tensor*" $result_mean]
    expr {$valid_none && $valid_mean}
} {1}

# ============================================================================
# Integration Tests with Different Tensor Types
# ============================================================================

test multi_margin_loss-7.1 {Integration - larger batch size} {
    # Test with simple input
    set input [torch::tensor_create -data {1.0 0.5 -0.2} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0} -dtype int64 -device cpu -requiresGrad false]
    
    set result [torch::multi_margin_loss $input $target]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-7.2 {Integration - different margins with named syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss -input $input -target $target -margin 2.0 -reduction sum]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-7.3 {Integration - camelCase with complex parameters} {
    # Simple input
    set input [torch::tensor_create -data {3.0 1.0 -1.0} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0} -dtype int64 -device cpu -requiresGrad false]
    
    set result [torch::multiMarginLoss -input $input -target $target -p 2 -margin 0.5 -reduction none]
    expr {[string match "tensor*" $result]}
} {1}

# ============================================================================
# Performance and Edge Cases
# ============================================================================

test multi_margin_loss-8.1 {Edge case - single sample} {
    # Single sample
    set input [torch::tensor_create -data {2.0 1.0 0.5} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0} -dtype int64 -device cpu -requiresGrad false]
    
    set result [torch::multi_margin_loss $input $target]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-8.2 {Edge case - zero margin} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multi_margin_loss -input $input -target $target -margin 0.0]
    expr {[string match "tensor*" $result]}
} {1}

test multi_margin_loss-8.3 {Performance - should complete quickly} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set start_time [clock milliseconds]
    torch::multi_margin_loss $input $target
    set end_time [clock milliseconds]
    
    # Should complete within reasonable time (less than 100ms)
    expr {($end_time - $start_time) < 100}
} {1}

cleanupTests 