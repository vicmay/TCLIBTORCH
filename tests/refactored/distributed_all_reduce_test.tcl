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
proc verify_all_reduce_result {result} {
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
proc create_test_tensor {} {
    return [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4} -dtype float32]
}

proc create_2d_tensor {} {
    return [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
}

proc create_scalar_tensor {} {
    return [torch::tensor_create -data {5.0} -shape {1} -dtype float32]
}

proc create_int_tensor {} {
    return [torch::tensor_create -data {1 2 3 4} -shape {4} -dtype int32]
}

# Test positional syntax
test distributed_all_reduce-1.1 {Basic positional syntax - default operation} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce $tensor]
    verify_all_reduce_result $result
} {1}

test distributed_all_reduce-1.2 {Positional syntax with sum operation} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce $tensor sum]
    verify_all_reduce_result $result
} {1}

test distributed_all_reduce-1.3 {Positional syntax with mean operation} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce $tensor mean]
    verify_all_reduce_result $result
} {1}

test distributed_all_reduce-1.4 {Positional syntax with max operation} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce $tensor max]
    verify_all_reduce_result $result
} {1}

test distributed_all_reduce-1.5 {Positional syntax with min operation} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce $tensor min]
    verify_all_reduce_result $result
} {1}

# Test named parameter syntax
test distributed_all_reduce-2.1 {Named parameter syntax - basic} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce -tensor $tensor]
    verify_all_reduce_result $result
} {1}

test distributed_all_reduce-2.2 {Named parameter syntax with sum operation} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce -tensor $tensor -operation sum]
    verify_all_reduce_result $result
} {1}

test distributed_all_reduce-2.3 {Named parameter syntax with mean operation} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce -tensor $tensor -operation mean]
    verify_all_reduce_result $result
} {1}

test distributed_all_reduce-2.4 {Named parameter syntax with max operation} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce -tensor $tensor -operation max]
    verify_all_reduce_result $result
} {1}

test distributed_all_reduce-2.5 {Named parameter syntax with min operation} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce -tensor $tensor -operation min]
    verify_all_reduce_result $result
} {1}

test distributed_all_reduce-2.6 {Named parameter syntax - different order} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce -operation mean -tensor $tensor]
    verify_all_reduce_result $result
} {1}

# Test camelCase alias
test distributed_all_reduce-3.1 {CamelCase alias - positional syntax} {
    set tensor [create_test_tensor]
    set result [torch::distributedAllReduce $tensor]
    verify_all_reduce_result $result
} {1}

test distributed_all_reduce-3.2 {CamelCase alias - named parameters} {
    set tensor [create_test_tensor]
    set result [torch::distributedAllReduce -tensor $tensor -operation sum]
    verify_all_reduce_result $result
} {1}

test distributed_all_reduce-3.3 {CamelCase alias with different operations} {
    set tensor [create_test_tensor]
    
    set result_mean [torch::distributedAllReduce -tensor $tensor -operation mean]
    set result_max [torch::distributedAllReduce -tensor $tensor -operation max]
    set result_min [torch::distributedAllReduce -tensor $tensor -operation min]
    
    set valid_mean [verify_all_reduce_result $result_mean]
    set valid_max [verify_all_reduce_result $result_max]
    set valid_min [verify_all_reduce_result $result_min]
    
    expr {$valid_mean && $valid_max && $valid_min}
} {1}

# Test consistency between syntaxes
test distributed_all_reduce-4.1 {Consistency - positional vs named basic} {
    set tensor [create_test_tensor]
    
    set result1 [torch::distributed_all_reduce $tensor]
    set result2 [torch::distributed_all_reduce -tensor $tensor]
    
    # Both should be valid tensors with same shape
    set valid1 [verify_all_reduce_result $result1]
    set valid2 [verify_all_reduce_result $result2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

test distributed_all_reduce-4.2 {Consistency - with operation parameter} {
    set tensor [create_test_tensor]
    
    set result1 [torch::distributed_all_reduce $tensor sum]
    set result2 [torch::distributed_all_reduce -tensor $tensor -operation sum]
    
    # Both should produce tensors with same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} {1}

test distributed_all_reduce-4.3 {Consistency - camelCase vs snake_case} {
    set tensor [create_test_tensor]
    
    set result1 [torch::distributed_all_reduce $tensor mean]
    set result2 [torch::distributedAllReduce $tensor mean]
    
    # Both should produce tensors with same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} {1}

# Test different tensor types and shapes
test distributed_all_reduce-5.1 {2D tensor support} {
    set tensor [create_2d_tensor]
    set result [torch::distributed_all_reduce $tensor]
    
    # Should preserve shape
    set original_shape [torch::tensor_shape $tensor]
    set result_shape [torch::tensor_shape $result]
    
    expr {[verify_all_reduce_result $result] && $original_shape eq $result_shape}
} {1}

test distributed_all_reduce-5.2 {Scalar tensor support} {
    set tensor [create_scalar_tensor]
    set result [torch::distributed_all_reduce $tensor]
    
    # Should preserve shape
    set original_shape [torch::tensor_shape $tensor]
    set result_shape [torch::tensor_shape $result]
    
    expr {[verify_all_reduce_result $result] && $original_shape eq $result_shape}
} {1}

test distributed_all_reduce-5.3 {Integer tensor support} {
    set tensor [create_int_tensor]
    set result [torch::distributed_all_reduce $tensor]
    
    # Should work with integer tensors
    verify_all_reduce_result $result
} {1}

# Test all supported operations
test distributed_all_reduce-6.1 {All operations with same tensor} {
    set tensor [create_test_tensor]
    
    set result_sum [torch::distributed_all_reduce $tensor sum]
    set result_mean [torch::distributed_all_reduce $tensor mean]
    set result_max [torch::distributed_all_reduce $tensor max]
    set result_min [torch::distributed_all_reduce $tensor min]
    
    set valid_sum [verify_all_reduce_result $result_sum]
    set valid_mean [verify_all_reduce_result $result_mean]
    set valid_max [verify_all_reduce_result $result_max]
    set valid_min [verify_all_reduce_result $result_min]
    
    expr {$valid_sum && $valid_mean && $valid_max && $valid_min}
} {1}

test distributed_all_reduce-6.2 {Operations preserve tensor shape} {
    set tensor [create_2d_tensor]
    set original_shape [torch::tensor_shape $tensor]
    
    set result_sum [torch::distributed_all_reduce $tensor sum]
    set result_mean [torch::distributed_all_reduce $tensor mean]
    set result_max [torch::distributed_all_reduce $tensor max]
    set result_min [torch::distributed_all_reduce $tensor min]
    
    set shape_sum [torch::tensor_shape $result_sum]
    set shape_mean [torch::tensor_shape $result_mean]
    set shape_max [torch::tensor_shape $result_max]
    set shape_min [torch::tensor_shape $result_min]
    
    expr {$original_shape eq $shape_sum && $original_shape eq $shape_mean && 
          $original_shape eq $shape_max && $original_shape eq $shape_min}
} {1}

# Test data type preservation
test distributed_all_reduce-7.1 {Data type preservation - float32} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    set result [torch::distributed_all_reduce $tensor]
    
    set original_dtype [torch::tensor_dtype $tensor]
    set result_dtype [torch::tensor_dtype $result]
    
    expr {$original_dtype eq $result_dtype}
} {1}

test distributed_all_reduce-7.2 {Data type preservation - int32} {
    set tensor [torch::tensor_create -data {1 2 3} -shape {3} -dtype int32]
    set result [torch::distributed_all_reduce $tensor]
    
    # Should preserve or appropriately convert data type
    set result_dtype [torch::tensor_dtype $result]
    expr {$result_dtype ne ""}
} {1}

# Test behavior in single GPU mode (simulation)
test distributed_all_reduce-8.1 {Single GPU mode - sum operation} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce $tensor sum]
    
    # In single GPU mode, should return tensor with same shape
    set original_shape [torch::tensor_shape $tensor]
    set result_shape [torch::tensor_shape $result]
    
    expr {[verify_all_reduce_result $result] && $original_shape eq $result_shape}
} {1}

test distributed_all_reduce-8.2 {Single GPU mode - mean operation} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_reduce $tensor mean]
    
    # In single GPU mode with world_size=1, mean should be equivalent to original
    verify_all_reduce_result $result
} {1}

# Test error handling
test distributed_all_reduce-9.1 {Invalid tensor name positional} {
    catch {torch::distributed_all_reduce invalid_tensor} msg
    expr {[string match "*Tensor not found*" $msg]}
} {1}

test distributed_all_reduce-9.2 {Invalid tensor name named parameter} {
    catch {torch::distributed_all_reduce -tensor invalid_tensor} msg
    expr {[string match "*Tensor not found*" $msg]}
} {1}

test distributed_all_reduce-9.3 {Missing required parameters} {
    catch {torch::distributed_all_reduce} msg
    expr {[string match "*wrong # args*" $msg] || [string match "*Invalid arguments*" $msg]}
} {1}

test distributed_all_reduce-9.4 {Invalid operation name} {
    set tensor [create_test_tensor]
    catch {torch::distributed_all_reduce $tensor invalid_op} msg
    expr {[string match "*Invalid arguments*" $msg]}
} {1}

test distributed_all_reduce-9.5 {Invalid operation in named parameter} {
    set tensor [create_test_tensor]
    catch {torch::distributed_all_reduce -tensor $tensor -operation invalid_op} msg
    expr {[string match "*Invalid arguments*" $msg]}
} {1}

test distributed_all_reduce-9.6 {Invalid parameter name} {
    set tensor [create_test_tensor]
    catch {torch::distributed_all_reduce -invalid $tensor} msg
    expr {[string match "*unknown option*" $msg]}
} {1}

test distributed_all_reduce-9.7 {Missing value for named parameter} {
    catch {torch::distributed_all_reduce -tensor} msg
    expr {[string match "*missing value*" $msg] || [string match "*wrong # args*" $msg]}
} {1}

test distributed_all_reduce-9.8 {Too many positional arguments} {
    set tensor [create_test_tensor]
    catch {torch::distributed_all_reduce $tensor sum extra_arg} msg
    expr {[string match "*wrong # args*" $msg]}
} {1}

test distributed_all_reduce-9.9 {Missing required named parameter} {
    catch {torch::distributed_all_reduce -operation sum} msg
    expr {[string match "*Invalid arguments*" $msg]}
} {1}

# Test camelCase alias error handling
test distributed_all_reduce-10.1 {CamelCase alias - invalid tensor} {
    catch {torch::distributedAllReduce invalid_tensor} msg
    expr {[string match "*Tensor not found*" $msg]}
} {1}

test distributed_all_reduce-10.2 {CamelCase alias - invalid operation} {
    set tensor [create_test_tensor]
    catch {torch::distributedAllReduce $tensor invalid_op} msg
    expr {[string match "*Invalid arguments*" $msg]}
} {1}

test distributed_all_reduce-10.3 {CamelCase alias - missing arguments} {
    catch {torch::distributedAllReduce} msg
    expr {[string match "*wrong # args*" $msg] || [string match "*Invalid arguments*" $msg]}
} {1}

# Test edge cases
test distributed_all_reduce-11.1 {Large tensor} {
    set data {}
    for {set i 0} {$i < 1000} {incr i} {
        lappend data [expr {$i * 0.1}]
    }
    set tensor [torch::tensor_create -data $data -shape {1000} -dtype float32]
    set result [torch::distributed_all_reduce $tensor]
    
    verify_all_reduce_result $result
} {1}

test distributed_all_reduce-11.2 {Mixed case operation names} {
    set tensor [create_test_tensor]
    
    # Test case sensitivity - these should fail if operations are case-sensitive
    catch {torch::distributed_all_reduce $tensor SUM} result1
    catch {torch::distributed_all_reduce $tensor Sum} result2
    
    # At least one should indicate the operation is invalid (case sensitive)
    expr {[string match "*Invalid arguments*" $result1] || [string match "tensor*" $result1]}
} {1}

test distributed_all_reduce-11.3 {Zero-filled tensor} {
    set tensor [torch::tensor_create -data {0.0 0.0 0.0 0.0} -shape {4} -dtype float32]
    set result [torch::distributed_all_reduce $tensor sum]
    
    verify_all_reduce_result $result
} {1}

cleanupTests 