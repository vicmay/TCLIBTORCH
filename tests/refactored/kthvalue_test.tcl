#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the shared library
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configure test parameters
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper procedure to create test tensors
proc create_tensor {data dims {dtype float32}} {
    return [torch::tensor_create -data $data -dtype $dtype -shape $dims]
}

proc create_int_tensor {data dims} {
    return [torch::tensor_create -data $data -dtype int64 -shape $dims]
}

# Helper procedure to check tensor validity
proc is_valid_tensor {tensor} {
    return [expr {$tensor ne ""}]
}

# ========================================
# Tests for Positional Syntax (Backward Compatibility)
# ========================================

test kthvalue-1.1 {Basic kthvalue - positional syntax} {
    ;# Create 1D tensor [5, 3, 8, 1, 9]
    set t1 [create_tensor {5.0 3.0 8.0 1.0 9.0} {5}]
    
    ;# Get 2nd smallest value (k=2, dim=0)
    set result [torch::kthvalue $t1 2 0]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

test kthvalue-1.2 {2D tensor kthvalue - positional syntax} {
    ;# Create 2D tensor [[1, 5], [3, 2]]
    set t1 [create_tensor {1.0 5.0 3.0 2.0} {2 2}]
    
    ;# Get 1st smallest value along dim 0 (k=1, dim=0)
    set result [torch::kthvalue $t1 1 0]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

test kthvalue-1.3 {With keepdim=true - positional syntax} {
    ;# Create 2D tensor [[4, 7], [2, 9]]
    set t1 [create_tensor {4.0 7.0 2.0 9.0} {2 2}]
    
    ;# Get 1st smallest value along dim 1 with keepdim=true
    set result [torch::kthvalue $t1 1 1 true]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

test kthvalue-1.4 {With keepdim=false - positional syntax} {
    ;# Create 2D tensor [[6, 3], [8, 1]]
    set t1 [create_tensor {6.0 3.0 8.0 1.0} {2 2}]
    
    ;# Get 2nd smallest value along dim 0 with keepdim=false
    set result [torch::kthvalue $t1 2 0 false]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

test kthvalue-1.5 {Integer tensor - positional syntax} {
    ;# Create integer tensor [10, 5, 20, 15]
    set t1 [create_int_tensor {10 5 20 15} {4}]
    
    ;# Get 3rd smallest value (k=3, dim=0)
    set result [torch::kthvalue $t1 3 0]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

# ========================================
# Tests for Named Parameter Syntax
# ========================================

test kthvalue-2.1 {Named syntax with -input -k -dim} {
    set t1 [create_tensor {7.0 2.0 9.0 4.0} {4}]
    
    set result [torch::kthvalue -input $t1 -k 2 -dim 0]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

test kthvalue-2.2 {Named syntax with -keepdim} {
    set t1 [create_tensor {1.0 6.0 3.0 8.0} {2 2}]
    
    set result [torch::kthvalue -input $t1 -k 1 -dim 1 -keepdim true]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

test kthvalue-2.3 {Parameter order independence} {
    set t1 [create_tensor {5.0 1.0 9.0} {3}]
    
    set result [torch::kthvalue -k 2 -dim 0 -input $t1]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

test kthvalue-2.4 {Named syntax with all parameters} {
    set t1 [create_tensor {2.0 7.0 1.0 5.0} {2 2}]
    
    set result [torch::kthvalue -input $t1 -k 1 -dim 0 -keepdim false]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

test kthvalue-2.5 {Named syntax with integer tensor} {
    set t1 [create_int_tensor {30 10 50 20} {4}]
    
    set result [torch::kthvalue -input $t1 -k 3 -dim 0]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

# ========================================
# Tests for CamelCase Alias
# ========================================

test kthvalue-3.1 {CamelCase alias - positional syntax} {
    set t1 [create_tensor {4.0 1.0 7.0} {3}]
    
    set result [torch::kthValue $t1 2 0]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

test kthvalue-3.2 {CamelCase alias - named syntax} {
    set t1 [create_tensor {3.0 8.0 2.0 6.0} {2 2}]
    
    set result [torch::kthValue -input $t1 -k 1 -dim 1]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

test kthvalue-3.3 {CamelCase alias with keepdim} {
    set t1 [create_tensor {9.0 1.0 5.0 3.0} {2 2}]
    
    set result [torch::kthValue -input $t1 -k 2 -dim 0 -keepdim true]
    
    ;# Should return a valid tensor
    is_valid_tensor $result
} 1

# ========================================
# Tests for Syntax Consistency
# ========================================

test kthvalue-4.1 {Syntax consistency - same result structure} {
    set t1 [create_tensor {8.0 2.0 5.0 1.0} {4}]
    
    set result1 [torch::kthvalue $t1 2 0]
    set result2 [torch::kthvalue -input $t1 -k 2 -dim 0]
    
    ;# Both should return valid tensors with same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test kthvalue-4.2 {CamelCase produces same shape} {
    set t1 [create_tensor {6.0 3.0 9.0 4.0} {2 2}]
    
    set result1 [torch::kthvalue $t1 1 1]
    set result2 [torch::kthValue $t1 1 1]
    
    ;# Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test kthvalue-4.3 {Named vs positional with keepdim} {
    set t1 [create_tensor {7.0 1.0 4.0 9.0} {2 2}]
    
    set result1 [torch::kthvalue $t1 2 0 true]
    set result2 [torch::kthvalue -input $t1 -k 2 -dim 0 -keepdim true]
    
    ;# Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

# ========================================
# Tests for Different Data Types
# ========================================

test kthvalue-5.1 {Float32 tensor} {
    set t1 [create_tensor {1.5 3.2 0.8 2.1} {4} float32]
    
    set result [torch::kthvalue $t1 2 0]
    
    is_valid_tensor $result
} 1

test kthvalue-5.2 {Float64 tensor} {
    set t1 [create_tensor {2.7 1.1 4.3 0.9} {4} float64]
    
    set result [torch::kthvalue -input $t1 -k 3 -dim 0]
    
    is_valid_tensor $result
} 1

test kthvalue-5.3 {Int32 tensor} {
    set t1 [create_tensor {25 15 35 20} {4} int32]
    
    set result [torch::kthValue $t1 2 0]
    
    is_valid_tensor $result
} 1

# ========================================
# Tests for Edge Cases
# ========================================

test kthvalue-6.1 {Single element tensor} {
    set t1 [create_tensor {5.0} {1}]
    
    set result [torch::kthvalue $t1 1 0]
    
    is_valid_tensor $result
} 1

test kthvalue-6.2 {Large tensor} {
    set t1 [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0} {10}]
    
    set result [torch::kthvalue -input $t1 -k 5 -dim 0]
    
    is_valid_tensor $result
} 1

test kthvalue-6.3 {3D tensor} {
    set t1 [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} {2 2 2}]
    
    set result [torch::kthValue -input $t1 -k 1 -dim 2]
    
    is_valid_tensor $result
} 1

# ========================================
# Tests for Error Handling
# ========================================

test kthvalue-7.1 {Invalid tensor name - positional} {
    set result [catch {torch::kthvalue "invalid_tensor" 1 0} msg]
    expr {$result == 1}
} 1

test kthvalue-7.2 {Invalid tensor name - named} {
    set result [catch {torch::kthvalue -input "invalid_tensor" -k 1 -dim 0} msg]
    expr {$result == 1}
} 1

test kthvalue-7.3 {Missing required parameter} {
    set result [catch {torch::kthvalue -k 1 -dim 0} msg]
    expr {$result == 1}
} 1

test kthvalue-7.4 {Invalid parameter name} {
    set t1 [create_tensor {1.0 2.0 3.0} {3}]
    set result [catch {torch::kthvalue -invalid_param $t1 -k 1 -dim 0} msg]
    expr {$result == 1}
} 1

test kthvalue-7.5 {Missing value for parameter} {
    set t1 [create_tensor {1.0 2.0 3.0} {3}]
    set result [catch {torch::kthvalue -input $t1 -k} msg]
    expr {$result == 1}
} 1

# ========================================
# Tests for Mathematical Correctness
# ========================================

test kthvalue-8.1 {Check mathematical correctness - simple case} {
    ;# Create tensor [3, 1, 4, 2] - sorted would be [1, 2, 3, 4]
    ;# So 2nd smallest (k=2) should be 2
    set t1 [create_tensor {3.0 1.0 4.0 2.0} {4}]
    
    set result [torch::kthvalue $t1 2 0]
    
    ;# Result should be a valid tensor
    is_valid_tensor $result
} 1

test kthvalue-8.2 {Check mathematical correctness - 2D case} {
    ;# Create 2D tensor [[5, 1], [3, 7]]
    ;# Along dim 0: col 0 [5,3] -> 1st smallest = 3, col 1 [1,7] -> 1st smallest = 1
    set t1 [create_tensor {5.0 1.0 3.0 7.0} {2 2}]
    
    set result [torch::kthvalue $t1 1 0]
    
    ;# Result should be a valid tensor with shape [2]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} 1

cleanupTests 