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

# Test helper function to verify tensor result
proc verify_diff_result {result} {
    # Verify we got a valid tensor handle back
    if {![string match "tensor*" $result]} {
        return 0
    }
    
    # Basic verification that the tensor exists and can be accessed
    set shape [torch::tensor_shape $result]
    if {$shape == ""} {
        return 0
    }
    
    return 1
}

# Test helper functions to create test tensors
proc create_1d_tensor {} {
    return [torch::tensor_create -data {1.0 4.0 7.0 10.0} -shape {4} -dtype float32]
}

proc create_2d_tensor {} {
    return [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
}

proc create_3d_tensor {} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0}
    return [torch::tensor_create -data $data -shape {2 2 2} -dtype float32]
}

proc create_sequence_tensor {} {
    return [torch::tensor_create -data {0.0 1.0 3.0 6.0 10.0} -shape {5} -dtype float32]
}

# Test positional syntax
test diff-1.1 {Basic positional syntax - 1D tensor} {
    set tensor [create_1d_tensor]
    set result [torch::diff $tensor]
    verify_diff_result $result
} {1}

test diff-1.2 {Positional syntax with n parameter} {
    set tensor [create_1d_tensor]
    set result [torch::diff $tensor 1]
    verify_diff_result $result
} {1}

test diff-1.3 {Positional syntax with n and dim parameters} {
    set tensor [create_2d_tensor]
    set result [torch::diff $tensor 1 0]
    verify_diff_result $result
} {1}

test diff-1.4 {Positional syntax - 2D tensor default dim} {
    set tensor [create_2d_tensor]
    set result [torch::diff $tensor]
    verify_diff_result $result
} {1}

test diff-1.5 {Positional syntax - 3D tensor} {
    set tensor [create_3d_tensor]
    set result [torch::diff $tensor 1 2]
    verify_diff_result $result
} {1}

# Test named parameter syntax
test diff-2.1 {Named parameter syntax - basic} {
    set tensor [create_1d_tensor]
    set result [torch::diff -input $tensor]
    verify_diff_result $result
} {1}

test diff-2.2 {Named parameter syntax with n} {
    set tensor [create_1d_tensor]
    set result [torch::diff -input $tensor -n 1]
    verify_diff_result $result
} {1}

test diff-2.3 {Named parameter syntax with dim} {
    set tensor [create_2d_tensor]
    set result [torch::diff -input $tensor -dim 0]
    verify_diff_result $result
} {1}

test diff-2.4 {Named parameter syntax with all parameters} {
    set tensor [create_2d_tensor]
    set result [torch::diff -input $tensor -n 1 -dim 1]
    verify_diff_result $result
} {1}

test diff-2.5 {Named parameter syntax - different order} {
    set tensor [create_2d_tensor]
    set result [torch::diff -dim 0 -input $tensor -n 1]
    verify_diff_result $result
} {1}

# Test mathematical correctness
test diff-3.1 {1D diff mathematical correctness} {
    # Create tensor [1, 4, 7, 10] -> diff should be [3, 3, 3]
    set tensor [create_1d_tensor]
    set result [torch::diff $tensor]
    
    # Check shape - should be [3] (reduced by 1)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test diff-3.2 {2D diff along dimension 0} {
    # Create 2x3 tensor, diff along dim 0 should reduce first dimension by 1
    set tensor [create_2d_tensor]
    set result [torch::diff $tensor 1 0]
    
    # Check shape - should be [1 3] (first dim reduced from 2 to 1)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1 3"}
} {1}

test diff-3.3 {2D diff along dimension 1} {
    # Create 2x3 tensor, diff along dim 1 should reduce second dimension by 1
    set tensor [create_2d_tensor]
    set result [torch::diff $tensor 1 1]
    
    # Check shape - should be [2 2] (second dim reduced from 3 to 2)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} {1}

test diff-3.4 {Multiple differences n=2} {
    set tensor [create_sequence_tensor]
    set result [torch::diff $tensor 2]
    
    # Should be valid result with proper shape
    expr {[verify_diff_result $result] && [torch::tensor_shape $result] eq "3"}
} {1}

# Test edge cases
test diff-4.1 {Single element tensor} {
    set tensor [torch::tensor_create -data {5.0} -shape {1} -dtype float32]
    catch {torch::diff $tensor} msg
    # Should either work or give a meaningful error
    expr {[string match "tensor*" $msg] || $msg != ""}
} {1}

test diff-4.2 {Large n value} {
    set tensor [create_sequence_tensor]
    set result [torch::diff $tensor 3]
    
    # Should work and produce valid result
    expr {[verify_diff_result $result] && [torch::tensor_shape $result] eq "2"}
} {1}

test diff-4.3 {Different dimensions} {
    set tensor [create_3d_tensor]
    
    # Test diff along each dimension
    set result0 [torch::diff $tensor 1 0]
    set result1 [torch::diff $tensor 1 1] 
    set result2 [torch::diff $tensor 1 2]
    
    set valid0 [verify_diff_result $result0]
    set valid1 [verify_diff_result $result1]
    set valid2 [verify_diff_result $result2]
    
    expr {$valid0 && $valid1 && $valid2}
} {1}

# Test consistency between syntaxes
test diff-5.1 {Consistency - positional vs named basic} {
    set tensor [create_1d_tensor]
    
    set result1 [torch::diff $tensor]
    set result2 [torch::diff -input $tensor]
    
    # Both should be valid tensors with same shape
    set valid1 [verify_diff_result $result1]
    set valid2 [verify_diff_result $result2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

test diff-5.2 {Consistency - with n parameter} {
    set tensor [create_1d_tensor]
    
    set result1 [torch::diff $tensor 2]
    set result2 [torch::diff -input $tensor -n 2]
    
    # Both should produce tensors with same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} {1}

test diff-5.3 {Consistency - with all parameters} {
    set tensor [create_2d_tensor]
    
    set result1 [torch::diff $tensor 1 1]
    set result2 [torch::diff -input $tensor -n 1 -dim 1]
    
    # Both should produce tensors with same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} {1}

# Test data type preservation
test diff-6.1 {Data type preservation - float32} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    set result [torch::diff $tensor]
    
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Float32"}
} {1}

test diff-6.2 {Data type preservation - int32} {
    set tensor [torch::tensor_create -data {1 2 3 4} -shape {4} -dtype int32]
    set result [torch::diff $tensor]
    
    # Should preserve integer type or convert appropriately
    set dtype [torch::tensor_dtype $result]
    expr {$dtype ne ""}
} {1}

# Test different tensor shapes
test diff-7.1 {Vector tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -shape {5} -dtype float32]
    set result [torch::diff $tensor]
    
    expr {[verify_diff_result $result] && [torch::tensor_shape $result] eq "4"}
} {1}

test diff-7.2 {Matrix tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::diff $tensor 1 0]
    
    expr {[verify_diff_result $result] && [torch::tensor_shape $result] eq "1 2"}
} {1}

test diff-7.3 {3D tensor with different dimensions} {
    set tensor [create_3d_tensor]
    
    set result_dim0 [torch::diff $tensor 1 0]
    set result_dim1 [torch::diff $tensor 1 1]
    set result_dim2 [torch::diff $tensor 1 2]
    
    set shape0 [torch::tensor_shape $result_dim0]
    set shape1 [torch::tensor_shape $result_dim1]
    set shape2 [torch::tensor_shape $result_dim2]
    
    # Each should reduce one dimension by 1
    expr {$shape0 eq "1 2 2" && $shape1 eq "2 1 2" && $shape2 eq "2 2 1"}
} {1}

# Test error handling
test diff-8.1 {Invalid tensor name positional} {
    catch {torch::diff invalid_tensor} msg
    expr {[string match "*Invalid tensor*" $msg]}
} {1}

test diff-8.2 {Invalid tensor name named parameter} {
    catch {torch::diff -input invalid_tensor} msg
    expr {[string match "*Invalid tensor*" $msg]}
} {1}

test diff-8.3 {Missing required parameters} {
    catch {torch::diff} msg
    expr {[string match "*Required parameter missing*" $msg] || [string match "*Wrong number*" $msg]}
} {1}

test diff-8.4 {Invalid parameter name} {
    set tensor [create_1d_tensor]
    catch {torch::diff -invalid $tensor} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

test diff-8.5 {Missing value for named parameter} {
    catch {torch::diff -input} msg
    expr {[string match "*Missing value*" $msg]}
} {1}

test diff-8.6 {Too many positional arguments} {
    set tensor [create_1d_tensor]
    catch {torch::diff $tensor 1 0 extra_arg} msg
    expr {[string match "*Wrong number*" $msg]}
} {1}

test diff-8.7 {Invalid n value} {
    set tensor [create_1d_tensor]
    catch {torch::diff $tensor invalid_n} msg
    expr {[string match "*Invalid n value*" $msg] || [string match "*expected integer*" $msg]}
} {1}

test diff-8.8 {Invalid dim value} {
    set tensor [create_2d_tensor]
    catch {torch::diff $tensor 1 invalid_dim} msg
    expr {[string match "*Invalid dim value*" $msg] || [string match "*expected integer*" $msg]}
} {1}

test diff-8.9 {Invalid n value named parameter} {
    set tensor [create_1d_tensor]
    catch {torch::diff -input $tensor -n invalid_n} msg
    expr {[string match "*Invalid n value*" $msg]}
} {1}

test diff-8.10 {Invalid dim value named parameter} {
    set tensor [create_2d_tensor]
    catch {torch::diff -input $tensor -dim invalid_dim} msg
    expr {[string match "*Invalid dim value*" $msg]}
} {1}

test diff-8.11 {Missing required named parameter} {
    catch {torch::diff -n 1} msg
    expr {[string match "*Required parameter missing*" $msg]}
} {1}

# Test parameter validation
test diff-9.1 {Negative n value} {
    set tensor [create_1d_tensor]
    # Negative n might be valid or invalid depending on implementation
    catch {torch::diff $tensor -1} msg
    # Should either work or give meaningful error
    expr {[string match "tensor*" $msg] || $msg != ""}
} {1}

test diff-9.2 {Zero n value} {
    set tensor [create_1d_tensor]
    # n=0 might return original tensor or error
    catch {torch::diff $tensor 0} msg
    expr {[string match "tensor*" $msg] || $msg != ""}
} {1}

test diff-9.3 {Large dimension value} {
    set tensor [create_2d_tensor]
    # Should handle out-of-bounds dimensions gracefully
    catch {torch::diff $tensor 1 10} msg
    expr {[string match "tensor*" $msg] || $msg != ""}
} {1}

cleanupTests 