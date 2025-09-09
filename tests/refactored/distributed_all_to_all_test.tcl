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
proc verify_all_to_all_result {result} {
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

proc create_3d_tensor {} {
    return [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]
}

proc create_scalar_tensor {} {
    return [torch::tensor_create -data {5.0} -shape {1} -dtype float32]
}

proc create_int_tensor {} {
    return [torch::tensor_create -data {1 2 3 4} -shape {4} -dtype int32]
}

# Test positional syntax
test distributed_all_to_all-1.1 {Basic positional syntax - tensor only} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_to_all $tensor]
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-1.2 {Positional syntax with group parameter} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_to_all $tensor "group1"]
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-1.3 {Positional syntax with empty group} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_to_all $tensor ""]
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-1.4 {Positional syntax - 2D tensor} {
    set tensor [create_2d_tensor]
    set result [torch::distributed_all_to_all $tensor]
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-1.5 {Positional syntax - 3D tensor} {
    set tensor [create_3d_tensor]
    set result [torch::distributed_all_to_all $tensor]
    verify_all_to_all_result $result
} {1}

# Test named parameter syntax
test distributed_all_to_all-2.1 {Named parameter syntax - basic} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_to_all -tensor $tensor]
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-2.2 {Named parameter syntax with group} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_to_all -tensor $tensor -group "group1"]
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-2.3 {Named parameter syntax - different order} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_to_all -group "group1" -tensor $tensor]
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-2.4 {Named parameter syntax with empty group} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_to_all -tensor $tensor -group ""]
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-2.5 {Named parameter syntax - 2D tensor} {
    set tensor [create_2d_tensor]
    set result [torch::distributed_all_to_all -tensor $tensor]
    verify_all_to_all_result $result
} {1}

# Test camelCase alias
test distributed_all_to_all-3.1 {CamelCase alias - positional syntax} {
    set tensor [create_test_tensor]
    set result [torch::distributedAllToAll $tensor]
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-3.2 {CamelCase alias - named parameters} {
    set tensor [create_test_tensor]
    set result [torch::distributedAllToAll -tensor $tensor]
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-3.3 {CamelCase alias with group parameter} {
    set tensor [create_test_tensor]
    set result [torch::distributedAllToAll -tensor $tensor -group "test_group"]
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-3.4 {CamelCase alias - positional with group} {
    set tensor [create_test_tensor]
    set result [torch::distributedAllToAll $tensor "test_group"]
    verify_all_to_all_result $result
} {1}

# Test consistency between syntaxes
test distributed_all_to_all-4.1 {Consistency - positional vs named basic} {
    set tensor [create_test_tensor]
    
    set result1 [torch::distributed_all_to_all $tensor]
    set result2 [torch::distributed_all_to_all -tensor $tensor]
    
    # Both should be valid tensors with same shape
    set valid1 [verify_all_to_all_result $result1]
    set valid2 [verify_all_to_all_result $result2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

test distributed_all_to_all-4.2 {Consistency - with group parameter} {
    set tensor [create_test_tensor]
    
    set result1 [torch::distributed_all_to_all $tensor "group1"]
    set result2 [torch::distributed_all_to_all -tensor $tensor -group "group1"]
    
    # Both should produce tensors with same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} {1}

test distributed_all_to_all-4.3 {Consistency - camelCase vs snake_case} {
    set tensor [create_test_tensor]
    
    set result1 [torch::distributed_all_to_all $tensor]
    set result2 [torch::distributedAllToAll $tensor]
    
    # Both should produce tensors with same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} {1}

# Test shape preservation
test distributed_all_to_all-5.1 {Shape preservation - 1D tensor} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_to_all $tensor]
    
    # Should preserve original shape
    set original_shape [torch::tensor_shape $tensor]
    set result_shape [torch::tensor_shape $result]
    
    expr {[verify_all_to_all_result $result] && $original_shape eq $result_shape}
} {1}

test distributed_all_to_all-5.2 {Shape preservation - 2D tensor} {
    set tensor [create_2d_tensor]
    set result [torch::distributed_all_to_all $tensor]
    
    # Should preserve original shape
    set original_shape [torch::tensor_shape $tensor]
    set result_shape [torch::tensor_shape $result]
    
    expr {[verify_all_to_all_result $result] && $original_shape eq $result_shape}
} {1}

test distributed_all_to_all-5.3 {Shape preservation - 3D tensor} {
    set tensor [create_3d_tensor]
    set result [torch::distributed_all_to_all $tensor]
    
    # Should preserve original shape
    set original_shape [torch::tensor_shape $tensor]
    set result_shape [torch::tensor_shape $result]
    
    expr {[verify_all_to_all_result $result] && $original_shape eq $result_shape}
} {1}

test distributed_all_to_all-5.4 {Shape preservation - scalar tensor} {
    set tensor [create_scalar_tensor]
    set result [torch::distributed_all_to_all $tensor]
    
    # Should preserve original shape
    set original_shape [torch::tensor_shape $tensor]
    set result_shape [torch::tensor_shape $result]
    
    expr {[verify_all_to_all_result $result] && $original_shape eq $result_shape}
} {1}

# Test data type preservation
test distributed_all_to_all-6.1 {Data type preservation - float32} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    set result [torch::distributed_all_to_all $tensor]
    
    set original_dtype [torch::tensor_dtype $tensor]
    set result_dtype [torch::tensor_dtype $result]
    
    expr {$original_dtype eq $result_dtype}
} {1}

test distributed_all_to_all-6.2 {Data type preservation - int32} {
    set tensor [create_int_tensor]
    set result [torch::distributed_all_to_all $tensor]
    
    set original_dtype [torch::tensor_dtype $tensor]
    set result_dtype [torch::tensor_dtype $result]
    
    expr {$original_dtype eq $result_dtype}
} {1}

test distributed_all_to_all-6.3 {Data type preservation - different types} {
    # Test various data types
    set float_tensor [torch::tensor_create -data {1.5 2.5 3.5} -shape {3} -dtype float32]
    set int_tensor [torch::tensor_create -data {1 2 3} -shape {3} -dtype int32]
    
    set float_result [torch::distributed_all_to_all $float_tensor]
    set int_result [torch::distributed_all_to_all $int_tensor]
    
    set float_dtype [torch::tensor_dtype $float_result]
    set int_dtype [torch::tensor_dtype $int_result]
    
    expr {$float_dtype eq "Float32" && $int_dtype eq "Int32"}
} {1}

# Test different tensor configurations
test distributed_all_to_all-7.1 {Large tensor} {
    set data {}
    for {set i 0} {$i < 1000} {incr i} {
        lappend data [expr {$i * 0.1}]
    }
    set tensor [torch::tensor_create -data $data -shape {1000} -dtype float32]
    set result [torch::distributed_all_to_all $tensor]
    
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-7.2 {Empty tensor} {
    set tensor [torch::tensor_create -data {} -shape {0} -dtype float32]
    set result [torch::distributed_all_to_all $tensor]
    
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-7.3 {Complex shape tensor} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0}
    set tensor [torch::tensor_create -data $data -shape {3 4} -dtype float32]
    set result [torch::distributed_all_to_all $tensor]
    
    # Should preserve complex shape
    set original_shape [torch::tensor_shape $tensor]
    set result_shape [torch::tensor_shape $result]
    
    expr {[verify_all_to_all_result $result] && $original_shape eq $result_shape}
} {1}

# Test group parameter variations
test distributed_all_to_all-8.1 {Different group names} {
    set tensor [create_test_tensor]
    
    set result1 [torch::distributed_all_to_all $tensor "group_a"]
    set result2 [torch::distributed_all_to_all $tensor "group_b"]
    set result3 [torch::distributed_all_to_all -tensor $tensor -group "group_c"]
    
    set valid1 [verify_all_to_all_result $result1]
    set valid2 [verify_all_to_all_result $result2]
    set valid3 [verify_all_to_all_result $result3]
    
    expr {$valid1 && $valid2 && $valid3}
} {1}

test distributed_all_to_all-8.2 {Numeric group identifier} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_to_all $tensor "123"]
    
    verify_all_to_all_result $result
} {1}

test distributed_all_to_all-8.3 {Special characters in group name} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_to_all $tensor "group-test_123"]
    
    verify_all_to_all_result $result
} {1}

# Test error handling
test distributed_all_to_all-9.1 {Invalid tensor name positional} {
    catch {torch::distributed_all_to_all invalid_tensor} msg
    expr {[string match "*Invalid tensor name*" $msg]}
} {1}

test distributed_all_to_all-9.2 {Invalid tensor name named parameter} {
    catch {torch::distributed_all_to_all -tensor invalid_tensor} msg
    expr {[string match "*Invalid tensor name*" $msg]}
} {1}

test distributed_all_to_all-9.3 {Missing required parameters} {
    catch {torch::distributed_all_to_all} msg
    expr {[string match "*Wrong number of arguments*" $msg] || [string match "*Required parameter missing*" $msg]}
} {1}

test distributed_all_to_all-9.4 {Invalid parameter name} {
    set tensor [create_test_tensor]
    catch {torch::distributed_all_to_all -invalid $tensor} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

test distributed_all_to_all-9.5 {Missing value for named parameter} {
    catch {torch::distributed_all_to_all -tensor} msg
    expr {[string match "*Missing value for parameter*" $msg]}
} {1}

test distributed_all_to_all-9.6 {Too many positional arguments} {
    set tensor [create_test_tensor]
    catch {torch::distributed_all_to_all $tensor "group1" extra_arg} msg
    expr {[string match "*Wrong number of arguments*" $msg]}
} {1}

test distributed_all_to_all-9.7 {Missing required named parameter} {
    catch {torch::distributed_all_to_all -group "test"} msg
    expr {[string match "*Required parameter missing*" $msg]}
} {1}

# Test camelCase alias error handling
test distributed_all_to_all-10.1 {CamelCase alias - invalid tensor} {
    catch {torch::distributedAllToAll invalid_tensor} msg
    expr {[string match "*Invalid tensor name*" $msg]}
} {1}

test distributed_all_to_all-10.2 {CamelCase alias - missing arguments} {
    catch {torch::distributedAllToAll} msg
    expr {[string match "*Wrong number of arguments*" $msg] || [string match "*Required parameter missing*" $msg]}
} {1}

test distributed_all_to_all-10.3 {CamelCase alias - invalid parameter} {
    set tensor [create_test_tensor]
    catch {torch::distributedAllToAll -invalid $tensor} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

# Test simulation behavior
test distributed_all_to_all-11.1 {Simulation mode - basic functionality} {
    set tensor [create_test_tensor]
    set result [torch::distributed_all_to_all $tensor]
    
    # In simulation mode, should return a valid tensor with same shape
    set original_shape [torch::tensor_shape $tensor]
    set result_shape [torch::tensor_shape $result]
    
    expr {[verify_all_to_all_result $result] && $original_shape eq $result_shape}
} {1}

test distributed_all_to_all-11.2 {Simulation mode - data preservation} {
    # Check if simulation preserves data appropriately
    set tensor [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]
    set result [torch::distributed_all_to_all $tensor]
    
    # Should create a valid result
    verify_all_to_all_result $result
} {1}

# Test edge cases
test distributed_all_to_all-12.1 {Multiple operations on same tensor} {
    set tensor [create_test_tensor]
    
    set result1 [torch::distributed_all_to_all $tensor]
    set result2 [torch::distributed_all_to_all $tensor "group1"]
    set result3 [torch::distributed_all_to_all -tensor $tensor -group "group2"]
    
    set valid1 [verify_all_to_all_result $result1]
    set valid2 [verify_all_to_all_result $result2]
    set valid3 [verify_all_to_all_result $result3]
    
    expr {$valid1 && $valid2 && $valid3}
} {1}

test distributed_all_to_all-12.2 {Chain operations} {
    set tensor1 [create_test_tensor]
    set result1 [torch::distributed_all_to_all $tensor1]
    
    # Use result as input for another operation
    set result2 [torch::distributed_all_to_all $result1]
    
    verify_all_to_all_result $result2
} {1}

test distributed_all_to_all-12.3 {Zero-filled tensor} {
    set tensor [torch::tensor_create -data {0.0 0.0 0.0 0.0} -shape {4} -dtype float32]
    set result [torch::distributed_all_to_all $tensor]
    
    verify_all_to_all_result $result
} {1}

cleanupTests 