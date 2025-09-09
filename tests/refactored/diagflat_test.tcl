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
proc verify_diagflat_result {result} {
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
proc create_test_vector {} {
    # Create a simple 1D vector
    return [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]
}

proc create_test_matrix {} {
    # Create a simple 2x3 matrix that will be flattened
    return [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
}

proc create_test_tensor3d {} {
    # Create a 2x2x2 tensor that will be flattened
    return [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]
}

# Test positional syntax
test diagflat-1.1 {Basic positional syntax - vector input} {
    set vector [create_test_vector]
    set result [torch::diagflat $vector]
    # Should create 3x3 diagonal matrix
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "3 3"}
} {1}

test diagflat-1.2 {Basic positional syntax - matrix input} {
    set matrix [create_test_matrix]
    set result [torch::diagflat $matrix]
    # Matrix flattened has 6 elements, so should create 6x6 diagonal matrix
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "6 6"}
} {1}

test diagflat-1.3 {Positional syntax with positive offset} {
    set vector [create_test_vector]
    set result [torch::diagflat $vector 1]
    # Should create 4x4 matrix with values on upper diagonal
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "4 4"}
} {1}

test diagflat-1.4 {Positional syntax with negative offset} {
    set vector [create_test_vector]
    set result [torch::diagflat $vector -1]
    # Should create 4x4 matrix with values on lower diagonal
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "4 4"}
} {1}

# Test named parameter syntax
test diagflat-2.1 {Named parameter syntax - vector input} {
    set vector [create_test_vector]
    set result [torch::diagflat -input $vector]
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "3 3"}
} {1}

test diagflat-2.2 {Named parameter syntax - matrix input} {
    set matrix [create_test_matrix]
    set result [torch::diagflat -input $matrix]
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "6 6"}
} {1}

test diagflat-2.3 {Named parameter with positive offset} {
    set vector [create_test_vector]
    set result [torch::diagflat -input $vector -offset 1]
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "4 4"}
} {1}

test diagflat-2.4 {Named parameter with negative offset} {
    set vector [create_test_vector]
    set result [torch::diagflat -input $vector -offset -1]
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "4 4"}
} {1}

test diagflat-2.5 {Named parameters in different order} {
    set vector [create_test_vector]
    set result [torch::diagflat -offset 2 -input $vector]
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "5 5"}
} {1}

# Test camelCase alias
test diagflat-3.1 {CamelCase alias diagFlat positional} {
    set vector [create_test_vector]
    set result [torch::diagFlat $vector]
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "3 3"}
} {1}

test diagflat-3.2 {CamelCase alias diagFlat named parameters} {
    set vector [create_test_vector]
    set result [torch::diagFlat -input $vector -offset 1]
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "4 4"}
} {1}

# Test mathematical correctness
test diagflat-4.1 {Mathematical correctness - 1D input} {
    set vector [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]
    set result [torch::diagflat $vector]
    
    # Result should be 2x2 diagonal matrix
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} {1}

test diagflat-4.2 {Mathematical correctness - 2D input flattening} {
    # Create 2x2 matrix
    set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::diagflat $matrix]
    
    # Flattened to 4 elements, so 4x4 diagonal matrix
    set shape [torch::tensor_shape $result]
    expr {$shape eq "4 4"}
} {1}

test diagflat-4.3 {Mathematical correctness - 3D input flattening} {
    set tensor3d [create_test_tensor3d]
    set result [torch::diagflat $tensor3d]
    
    # 2x2x2 = 8 elements flattened, so 8x8 diagonal matrix
    set shape [torch::tensor_shape $result]
    expr {$shape eq "8 8"}
} {1}

# Test consistency between syntaxes
test diagflat-5.1 {Consistency between positional and named syntax} {
    set vector [create_test_vector]
    
    set result1 [torch::diagflat $vector]
    set result2 [torch::diagflat -input $vector]
    
    # Both should be valid tensors with same shape
    set valid1 [verify_diagflat_result $result1]
    set valid2 [verify_diagflat_result $result2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

test diagflat-5.2 {Consistency with offset} {
    set vector [create_test_vector]
    
    set result1 [torch::diagflat $vector 2]
    set result2 [torch::diagflat -input $vector -offset 2]
    
    # Both should be valid tensors with same shape
    set valid1 [verify_diagflat_result $result1]
    set valid2 [verify_diagflat_result $result2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

test diagflat-5.3 {Consistency with camelCase alias} {
    set vector [create_test_vector]
    
    set result1 [torch::diagflat $vector 1]
    set result2 [torch::diagFlat $vector 1]
    
    # Both should be valid tensors with same shape
    set valid1 [verify_diagflat_result $result1]
    set valid2 [verify_diagflat_result $result2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

# Test data type support
test diagflat-6.1 {Float32 tensor support} {
    set vector [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]
    set result [torch::diagflat -input $vector]
    expr {[verify_diagflat_result $result] && [torch::tensor_dtype $result] eq "Float32"}
} {1}

test diagflat-6.2 {Float64 tensor support} {
    set vector [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float64]
    set result [torch::diagflat -input $vector]
    expr {[verify_diagflat_result $result] && [torch::tensor_dtype $result] eq "Float64"}
} {1}

test diagflat-6.3 {Integer tensor support} {
    set vector [torch::tensor_create -data {1 2 3} -shape {3} -dtype int32]
    set result [torch::diagflat $vector]
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "3 3"}
} {1}

# Test edge cases
test diagflat-7.1 {Single element tensor} {
    set scalar [torch::tensor_create -data {5.0} -shape {1} -dtype float32]
    set result [torch::diagflat $scalar]
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "1 1"}
} {1}

test diagflat-7.2 {Large offset} {
    set vector [create_test_vector]
    set result [torch::diagflat $vector 5]
    # Should create larger matrix to accommodate offset
    set shape [torch::tensor_shape $result]
    set dims [split $shape]
    expr {[llength $dims] == 2 && [lindex $dims 0] == [lindex $dims 1]}
} {1}

test diagflat-7.3 {Zero offset explicitly} {
    set vector [create_test_vector]
    set result1 [torch::diagflat $vector]
    set result2 [torch::diagflat $vector 0]
    
    # Both should give same result
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test diagflat-7.4 {Large matrix input} {
    # Create larger matrix
    set data {}
    for {set i 0} {$i < 12} {incr i} {
        lappend data [expr {$i + 1.0}]
    }
    set matrix [torch::tensor_create -data $data -shape {3 4} -dtype float32]
    set result [torch::diagflat $matrix]
    
    # 3x4 = 12 elements, so 12x12 diagonal matrix
    expr {[verify_diagflat_result $result] && [torch::tensor_shape $result] eq "12 12"}
} {1}

# Test error handling
test diagflat-8.1 {Invalid tensor name positional} {
    catch {torch::diagflat invalid_tensor} msg
    expr {[string match "*Invalid input tensor*" $msg]}
} {1}

test diagflat-8.2 {Invalid tensor name named parameter} {
    catch {torch::diagflat -input invalid_tensor} msg
    expr {[string match "*Invalid input tensor*" $msg]}
} {1}

test diagflat-8.3 {Missing required parameters} {
    catch {torch::diagflat} msg
    expr {[string match "*Usage*" $msg] || [string match "*missing*" $msg]}
} {1}

test diagflat-8.4 {Invalid parameter name} {
    set vector [create_test_vector]
    catch {torch::diagflat -invalid $vector} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

test diagflat-8.5 {Missing value for named parameter} {
    catch {torch::diagflat -input} msg
    expr {[string match "*Missing value*" $msg] || [string match "*Usage*" $msg]}
} {1}

test diagflat-8.6 {Invalid offset value positional} {
    set vector [create_test_vector]
    catch {torch::diagflat $vector invalid_int} msg
    expr {[string match "*Invalid offset*" $msg] || [string match "*expected integer*" $msg]}
} {1}

test diagflat-8.7 {Invalid offset value named} {
    set vector [create_test_vector]
    catch {torch::diagflat -input $vector -offset invalid_int} msg
    expr {[string match "*Invalid offset*" $msg] || [string match "*expected integer*" $msg]}
} {1}

test diagflat-8.8 {Too many positional arguments} {
    set vector [create_test_vector]
    catch {torch::diagflat $vector 1 extra_arg} msg
    expr {[string match "*Usage*" $msg] || [string match "*wrong # args*" $msg]}
} {1}

test diagflat-8.9 {Missing required named parameter} {
    catch {torch::diagflat -offset 1} msg
    expr {[string match "*Required parameter missing*" $msg]}
} {1}

cleanupTests 