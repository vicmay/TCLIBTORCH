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
# Test torch::distributed_scatter - Positional Syntax
# ============================================================================

test distributed_scatter-1.1 {Basic positional syntax - default src} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_scatter $tensor]
    
    # Should return a tensor (scatter with default src 0)
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-1.2 {Positional syntax with explicit src} {
    set tensor [create_test_tensor {3 3} float32]
    set result [torch::distributed_scatter $tensor 1]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-1.3 {Positional syntax with different src values} {
    set tensor [create_test_tensor {2 3} float32]
    set result [torch::distributed_scatter $tensor 2]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-1.4 {Positional syntax with group parameter} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_scatter $tensor 0 "group1"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-1.5 {Positional syntax with different group} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_scatter $tensor 1 "workers"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

# ============================================================================
# Test torch::distributed_scatter - Named Syntax
# ============================================================================

test distributed_scatter-2.1 {Named syntax with required tensor parameter} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_scatter -tensor $tensor]
    
    # Should return a tensor (default src 0)
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-2.2 {Named syntax with tensor and src parameters} {
    set tensor [create_test_tensor {3 3} float32]
    set result [torch::distributed_scatter -tensor $tensor -src 1]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-2.3 {Named syntax with different src values} {
    set tensor [create_test_tensor {2 3} float32]
    set result [torch::distributed_scatter -tensor $tensor -src 2]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-2.4 {Named syntax with all parameters} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_scatter -tensor $tensor -src 0 -group "group1"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-2.5 {Named syntax with parameters in different order} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_scatter -src 1 -tensor $tensor -group "workers"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-2.6 {Named syntax with zero src} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_scatter -tensor $tensor -src 0]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-2.7 {Named syntax with large src value} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributed_scatter -tensor $tensor -src 100]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

# ============================================================================
# Test torch::distributedScatter - camelCase Alias
# ============================================================================

test distributed_scatter-3.1 {camelCase alias - positional syntax} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributedScatter $tensor]
    
    # Should return a tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-3.2 {camelCase alias - positional with src} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributedScatter $tensor 1]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-3.3 {camelCase alias - named syntax} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributedScatter -tensor $tensor -src 1]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-3.4 {camelCase alias - all parameters} {
    set tensor [create_test_tensor {2 2} float32]
    set result [torch::distributedScatter -tensor $tensor -src 0 -group "group1"]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

# ============================================================================
# Test Command Existence
# ============================================================================

test distributed_scatter-4.1 {Verify torch::distributed_scatter command exists} {
    info commands torch::distributed_scatter
} {::torch::distributed_scatter}

test distributed_scatter-4.2 {Verify torch::distributedScatter camelCase alias exists} {
    info commands torch::distributedScatter
} {::torch::distributedScatter}

# ============================================================================
# Test Error Handling
# ============================================================================

test distributed_scatter-5.1 {Error handling - missing tensor argument} {
    catch {torch::distributed_scatter} error
    string match "*tensor*" $error
} {1}

test distributed_scatter-5.2 {Error handling - invalid tensor handle} {
    catch {torch::distributed_scatter "invalid_tensor"} error
    string match "*Invalid tensor handle*" $error
} {1}

test distributed_scatter-5.3 {Error handling - invalid src parameter} {
    set tensor [create_test_tensor {2 2} float32]
    catch {torch::distributed_scatter $tensor "invalid_src"} error
    string match "*Invalid src parameter*" $error
} {1}

test distributed_scatter-5.4 {Error handling - negative src parameter} {
    set tensor [create_test_tensor {2 2} float32]
    catch {torch::distributed_scatter $tensor -1} error
    string match "*invalid src parameter*" $error
} {1}

test distributed_scatter-5.5 {Error handling - named syntax missing tensor} {
    catch {torch::distributed_scatter -src 0} error
    string match "*tensor*" $error
} {1}

test distributed_scatter-5.6 {Error handling - named syntax invalid src} {
    set tensor [create_test_tensor {2 2} float32]
    catch {torch::distributed_scatter -tensor $tensor -src "invalid"} error
    string match "*Invalid -src parameter*" $error
} {1}

test distributed_scatter-5.7 {Error handling - named syntax negative src} {
    set tensor [create_test_tensor {2 2} float32]
    catch {torch::distributed_scatter -tensor $tensor -src -1} error
    string match "*invalid src parameter*" $error
} {1}

test distributed_scatter-5.8 {Error handling - unknown parameter} {
    set tensor [create_test_tensor {2 2} float32]
    catch {torch::distributed_scatter -tensor $tensor -unknown "value"} error
    string match "*Unknown parameter*" $error
} {1}

test distributed_scatter-5.9 {Error handling - missing value for parameter} {
    set tensor [create_test_tensor {2 2} float32]
    catch {torch::distributed_scatter -tensor $tensor -src} error
    string match "*Missing value*" $error
} {1}

# ============================================================================
# Test Syntax Equivalence
# ============================================================================

test distributed_scatter-6.1 {Syntax equivalence - positional vs named} {
    set tensor1 [create_test_tensor {2 2} float32]
    set tensor2 [create_test_tensor {2 2} float32]
    
    # Same data in both tensors
    set result1 [torch::distributed_scatter $tensor1 1]
    set result2 [torch::distributed_scatter -tensor $tensor2 -src 1]
    
    # Both should produce similar results (same shape)
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 == $shape2}
} {1}

test distributed_scatter-6.2 {Syntax equivalence - snake_case vs camelCase} {
    set tensor1 [create_test_tensor {2 2} float32]
    set tensor2 [create_test_tensor {2 2} float32]
    
    # Same data in both tensors
    set result1 [torch::distributed_scatter $tensor1 2]
    set result2 [torch::distributedScatter $tensor2 2]
    
    # Both should produce similar results
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 == $shape2}
} {1}

test distributed_scatter-6.3 {Syntax equivalence - default vs explicit src 0} {
    set tensor1 [create_test_tensor {2 2} float32]
    set tensor2 [create_test_tensor {2 2} float32]
    
    # Same data in both tensors - default src vs explicit src 0
    set result1 [torch::distributed_scatter $tensor1]
    set result2 [torch::distributed_scatter $tensor2 0]
    
    # Both should produce similar results
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 == $shape2}
} {1}

# ============================================================================
# Test Different Tensor Types and Shapes
# ============================================================================

test distributed_scatter-7.1 {Different tensor shapes - 1D tensor} {
    set tensor [create_test_tensor {4} float32]
    set result [torch::distributed_scatter $tensor 0]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-7.2 {Different tensor shapes - 3D tensor} {
    set tensor [create_test_tensor {2 2 2} float32]
    set result [torch::distributed_scatter $tensor 1]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-7.3 {Different tensor shapes - large tensor} {
    set tensor [create_test_tensor {4 4} float32]
    set result [torch::distributed_scatter $tensor 2]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

test distributed_scatter-7.4 {Different tensor shapes - rectangular tensor} {
    set tensor [create_test_tensor {3 4} float32]
    set result [torch::distributed_scatter -tensor $tensor -src 0]
    
    # Should return a valid tensor
    expr {[string match "tensor*" $result] && [llength [torch::tensor_shape $result]] > 0}
} {1}

# ============================================================================
# Test Operation Correctness
# ============================================================================

test distributed_scatter-8.1 {Operation correctness - preserves tensor shape} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::distributed_scatter $tensor 0]
    
    # Should return a tensor with same shape
    set shape [torch::tensor_shape $result]
    expr {$shape == {2 2}}
} {1}

test distributed_scatter-8.2 {Operation correctness - different src values produce valid results} {
    set tensor [torch::tensor_create -data {2.0 4.0 6.0 8.0} -shape {2 2} -dtype float32]
    
    set result1 [torch::distributed_scatter $tensor 0]
    set result2 [torch::distributed_scatter $tensor 1]
    set result3 [torch::distributed_scatter $tensor 2]
    
    # All should return tensors with same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set shape3 [torch::tensor_shape $result3]
    expr {$shape1 == {2 2} && $shape2 == {2 2} && $shape3 == {2 2}}
} {1}

test distributed_scatter-8.3 {Operation correctness - named vs positional same result shape} {
    set tensor [torch::tensor_create -data {1.0 5.0 2.0 8.0} -shape {2 2} -dtype float32]
    
    set result_pos [torch::distributed_scatter $tensor 1]
    set result_named [torch::distributed_scatter -tensor $tensor -src 1]
    
    # Should return tensors with same shape
    set shape_pos [torch::tensor_shape $result_pos]
    set shape_named [torch::tensor_shape $result_named]
    expr {$shape_pos == $shape_named}
} {1}

# ============================================================================
# Test Parameter Validation
# ============================================================================

test distributed_scatter-9.1 {Parameter validation - src must be non-negative} {
    set tensor [create_test_tensor {2 2} float32]
    
    # Test valid src values
    set result1 [torch::distributed_scatter $tensor 0]
    set result2 [torch::distributed_scatter $tensor 1]
    set result3 [torch::distributed_scatter $tensor 100]
    
    # All should succeed
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2] && [string match "tensor*" $result3]}
} {1}

test distributed_scatter-9.2 {Parameter validation - group parameter accepts any string} {
    set tensor [create_test_tensor {2 2} float32]
    
    # Test various group values
    set result1 [torch::distributed_scatter $tensor 0 ""]
    set result2 [torch::distributed_scatter $tensor 0 "group1"]
    set result3 [torch::distributed_scatter $tensor 0 "workers"]
    
    # All should succeed
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2] && [string match "tensor*" $result3]}
} {1}

test distributed_scatter-9.3 {Parameter validation - named syntax parameter order independence} {
    set tensor [create_test_tensor {2 2} float32]
    
    # Test different parameter orders
    set result1 [torch::distributed_scatter -tensor $tensor -src 1 -group "test"]
    set result2 [torch::distributed_scatter -src 1 -tensor $tensor -group "test"]
    set result3 [torch::distributed_scatter -group "test" -tensor $tensor -src 1]
    
    # All should succeed and produce same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set shape3 [torch::tensor_shape $result3]
    expr {$shape1 == $shape2 && $shape2 == $shape3}
} {1}

cleanupTests 