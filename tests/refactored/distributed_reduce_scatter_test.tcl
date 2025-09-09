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
# Helper Functions
# ============================================================================

proc create_test_tensor {shape dtype} {
    set data [list]
    set total_elements 1
    foreach dim $shape {
        set total_elements [expr $total_elements * $dim]
    }
    
    for {set i 0} {$i < $total_elements} {incr i} {
        lappend data [expr $i + 1]
    }
    
    return [torch::tensor_create -data $data -shape $shape -dtype $dtype]
}

# ============================================================================
# Test torch::distributed_reduce_scatter - Positional Syntax
# ============================================================================

test distributed_reduce_scatter-1.1 {Basic positional syntax - default sum operation} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_reduce_scatter $tensor]
    
    # Should return a tensor (reduce-scatter with default sum operation)
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-1.2 {Positional syntax with explicit sum operation} {
    set tensor [create_test_tensor {3 3} float32]
    set result [torch::distributed_reduce_scatter $tensor "sum"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-1.3 {Positional syntax with mean operation} {
    set tensor [create_test_tensor {2 3} float32]
    set result [torch::distributed_reduce_scatter $tensor "mean"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-1.4 {Positional syntax with max operation} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_reduce_scatter $tensor "max"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-1.5 {Positional syntax with min operation} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_reduce_scatter $tensor "min"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-1.6 {Positional syntax with product operation} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_reduce_scatter $tensor "product"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-1.7 {Positional syntax with group parameter} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_reduce_scatter $tensor "sum" "group1"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

# ============================================================================
# Test torch::distributed_reduce_scatter - Named Syntax
# ============================================================================

test distributed_reduce_scatter-2.1 {Named syntax with required tensor parameter} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_reduce_scatter -tensor $tensor]
    
    # Should return a tensor (default sum operation)
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-2.2 {Named syntax with tensor and op parameters} {
    set tensor [create_test_tensor {3 3} float32]
    set result [torch::distributed_reduce_scatter -tensor $tensor -op "sum"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-2.3 {Named syntax with mean operation} {
    set tensor [create_test_tensor {2 3} float32]
    set result [torch::distributed_reduce_scatter -tensor $tensor -op "mean"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-2.4 {Named syntax with max operation} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_reduce_scatter -tensor $tensor -op "max"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-2.5 {Named syntax with min operation} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_reduce_scatter -tensor $tensor -op "min"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-2.6 {Named syntax with product operation} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_reduce_scatter -tensor $tensor -op "product"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-2.7 {Named syntax with all parameters} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_reduce_scatter -tensor $tensor -op "sum" -group "group1"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-2.8 {Named syntax with parameters in different order} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_reduce_scatter -op "mean" -tensor $tensor -group "group1"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

# ============================================================================
# Test torch::distributedReduceScatter - camelCase Alias
# ============================================================================

test distributed_reduce_scatter-3.1 {camelCase alias - positional syntax} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributedReduceScatter $tensor]
    
    # Should return a tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-3.2 {camelCase alias - named syntax} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributedReduceScatter -tensor $tensor -op "sum"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-3.3 {camelCase alias - all parameters} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributedReduceScatter -tensor $tensor -op "mean" -group "group1"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

# ============================================================================
# Test Command Existence
# ============================================================================

test distributed_reduce_scatter-4.1 {Verify torch::distributed_reduce_scatter command exists} {
    info commands torch::distributed_reduce_scatter
} {::torch::distributed_reduce_scatter}

test distributed_reduce_scatter-4.2 {Verify torch::distributedReduceScatter camelCase alias exists} {
    info commands torch::distributedReduceScatter
} {::torch::distributedReduceScatter}

# ============================================================================
# Test Error Handling
# ============================================================================

test distributed_reduce_scatter-5.1 {Error handling - missing tensor argument} {
    catch {torch::distributed_reduce_scatter} error
    string match "*tensor*" $error
} {1}

test distributed_reduce_scatter-5.2 {Error handling - invalid tensor handle} {
    catch {torch::distributed_reduce_scatter "invalid_tensor"} error
    string match "*Invalid tensor handle*" $error
} {1}

test distributed_reduce_scatter-5.3 {Error handling - invalid operation} {
    set tensor [create_test_tensor {2 2} float32]
    catch {torch::distributed_reduce_scatter $tensor "invalid_op"} error
    string match "*invalid operation*" $error
} {1}

test distributed_reduce_scatter-5.4 {Error handling - named syntax missing tensor} {
    catch {torch::distributed_reduce_scatter -op "sum"} error
    string match "*tensor*" $error
} {1}

test distributed_reduce_scatter-5.5 {Error handling - named syntax invalid operation} {
    set tensor [create_test_tensor {2 2} float32]
    catch {torch::distributed_reduce_scatter -tensor $tensor -op "invalid"} error
    string match "*invalid operation*" $error
} {1}

test distributed_reduce_scatter-5.6 {Error handling - unknown parameter} {
    set tensor [create_test_tensor {2 2} float32]
    catch {torch::distributed_reduce_scatter -tensor $tensor -unknown "value"} error
    string match "*Unknown parameter*" $error
} {1}

test distributed_reduce_scatter-5.7 {Error handling - missing value for parameter} {
    set tensor [create_test_tensor {2 2} float32]
    catch {torch::distributed_reduce_scatter -tensor $tensor -op} error
    string match "*Missing value*" $error
} {1}

# ============================================================================
# Test Syntax Equivalence
# ============================================================================

test distributed_reduce_scatter-6.1 {Syntax equivalence - positional vs named} {
    set tensor1 [create_test_tensor {2 2} float32]
    set tensor2 [create_test_tensor {2 2} float32]
    
    # Same data in both tensors
    set result1 [torch::distributed_reduce_scatter $tensor1 "sum"]
    set result2 [torch::distributed_reduce_scatter -tensor $tensor2 -op "sum"]
    
    # Both should produce similar results (same shape)
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 == $shape2}
} {1}

test distributed_reduce_scatter-6.2 {Syntax equivalence - snake_case vs camelCase} {
    set tensor1 [create_test_tensor {2 2} float32]
    set tensor2 [create_test_tensor {2 2} float32]
    
    # Same data in both tensors
    set result1 [torch::distributed_reduce_scatter $tensor1 "mean"]
    set result2 [torch::distributedReduceScatter $tensor2 "mean"]
    
    # Both should produce similar results
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 == $shape2}
} {1}

# ============================================================================
# Test Different Tensor Types and Shapes
# ============================================================================

test distributed_reduce_scatter-7.1 {Different tensor shapes - 1D tensor} {
    set tensor [create_test_tensor {4} float32]
    set result [torch::distributed_reduce_scatter $tensor "sum"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-7.2 {Different tensor shapes - 3D tensor} {
    set tensor [create_test_tensor {2 2 2} float32]
    set result [torch::distributed_reduce_scatter $tensor "mean"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_reduce_scatter-7.3 {Different tensor shapes - large tensor} {
    set tensor [create_test_tensor {4 4} float32]
    set result [torch::distributed_reduce_scatter $tensor "max"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

# ============================================================================
# Test Operation Correctness
# ============================================================================

test distributed_reduce_scatter-8.1 {Operation correctness - sum operation produces valid result} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::distributed_reduce_scatter $tensor "sum"]
    
    # Should return a tensor with same shape
    set shape [torch::tensor_shape $result]
    expr {$shape == {2 2}}
} {1}

test distributed_reduce_scatter-8.2 {Operation correctness - mean operation produces valid result} {
    set tensor [torch::tensor_create -data {2.0 4.0 6.0 8.0} -shape {2 2} -dtype float32]
    set result [torch::distributed_reduce_scatter $tensor "mean"]
    
    # Should return a tensor with same shape  
    set shape [torch::tensor_shape $result]
    expr {$shape == {2 2}}
} {1}

test distributed_reduce_scatter-8.3 {Operation correctness - max operation produces valid result} {
    set tensor [torch::tensor_create -data {1.0 5.0 2.0 8.0} -shape {2 2} -dtype float32]
    set result [torch::distributed_reduce_scatter $tensor "max"]
    
    # Should return a tensor with same shape
    set shape [torch::tensor_shape $result]
    expr {$shape == {2 2}}
} {1}

cleanupTests 