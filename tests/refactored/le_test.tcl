#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Helper procedure to create test tensors
proc create_tensor {data dims {dtype float32}} {
    return [torch::tensor_create -data $data -dtype $dtype -shape $dims]
}

;# Test cases for positional syntax
test le-1.1 {Basic positional syntax - less than or equal comparison} -body {
    ;# Create test tensors
    set tensor1 [create_tensor {1.0 2.0 3.0} {3}]
    set tensor2 [create_tensor {1.5 2.0 2.5} {3}]
    
    ;# Test positional syntax
    set result [torch::le $tensor1 $tensor2]
    
    ;# Should return a boolean tensor with shape [3]
    set shape [torch::tensor_shape $result]
    
    if {$shape eq "3"} {
        return "success"
    } else {
        return "failed: expected shape 3, got $shape"
    }
} -result "success"

test le-1.2 {Positional syntax - equal values} -body {
    ;# Create test tensors with equal values
    set tensor1 [create_tensor {2.0 3.0 4.0} {3}]
    set tensor2 [create_tensor {2.0 3.0 4.0} {3}]
    
    ;# Test positional syntax
    set result [torch::le $tensor1 $tensor2]
    
    ;# Should return boolean tensor with shape [3]
    set shape [torch::tensor_shape $result]
    
    if {$shape eq "3"} {
        return "success"
    } else {
        return "failed: expected shape 3, got $shape"
    }
} -result "success"

;# Test cases for named parameter syntax
test le-2.1 {Named parameter syntax - input1/input2} -body {
    ;# Create test tensors
    set tensor1 [create_tensor {0.5 1.5 2.5} {3}]
    set tensor2 [create_tensor {1.0 1.5 2.0} {3}]
    
    ;# Test named parameter syntax
    set result [torch::le -input1 $tensor1 -input2 $tensor2]
    
    ;# Should return boolean tensor with shape [3]
    set shape [torch::tensor_shape $result]
    
    if {$shape eq "3"} {
        return "success"
    } else {
        return "failed: expected shape 3, got $shape"
    }
} -result "success"

test le-2.2 {Named parameter syntax - tensor1/tensor2 aliases} -body {
    ;# Create test tensors
    set tensor1 [create_tensor {1.0 2.0} {2}]
    set tensor2 [create_tensor {1.0 3.0} {2}]
    
    ;# Test named parameter syntax with tensor aliases
    set result [torch::le -tensor1 $tensor1 -tensor2 $tensor2]
    
    ;# Should return boolean tensor with shape [2]
    set shape [torch::tensor_shape $result]
    
    if {$shape eq "2"} {
        return "success"
    } else {
        return "failed: expected shape 2, got $shape"
    }
} -result "success"

;# Test cases for camelCase alias
test le-3.1 {camelCase alias - positional syntax} -body {
    ;# Create test tensors
    set tensor1 [create_tensor {1.0 4.0} {2}]
    set tensor2 [create_tensor {2.0 3.0} {2}]
    
    ;# Test camelCase alias with positional syntax
    set result [torch::Le $tensor1 $tensor2]
    
    ;# Should return boolean tensor with shape [2]
    set shape [torch::tensor_shape $result]
    
    if {$shape eq "2"} {
        return "success"
    } else {
        return "failed: expected shape 2, got $shape"
    }
} -result "success"

test le-3.2 {camelCase alias - named parameter syntax} -body {
    ;# Create test tensors
    set tensor1 [create_tensor {2.5 3.5} {2}]
    set tensor2 [create_tensor {2.5 4.0} {2}]
    
    ;# Test camelCase alias with named parameter syntax
    set result [torch::Le -input1 $tensor1 -input2 $tensor2]
    
    ;# Should return boolean tensor with shape [2]
    set shape [torch::tensor_shape $result]
    
    if {$shape eq "2"} {
        return "success"
    } else {
        return "failed: expected shape 2, got $shape"
    }
} -result "success"

;# Test cases for parameter validation
test le-4.1 {Parameter validation - both syntaxes produce same result} -body {
    ;# Create identical test tensors
    set tensor1a [create_tensor {1.0 2.0 3.0} {3}]
    set tensor2a [create_tensor {1.5 2.0 2.5} {3}]
    set tensor1b [create_tensor {1.0 2.0 3.0} {3}]
    set tensor2b [create_tensor {1.5 2.0 2.5} {3}]
    
    ;# Test both syntaxes
    set result1 [torch::le $tensor1a $tensor2a]
    set result2 [torch::le -input1 $tensor1b -input2 $tensor2b]
    
    ;# Both should return same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    if {$shape1 eq $shape2} {
        return "success"
    } else {
        return "failed: different shapes from both syntaxes"
    }
} -result "success"

;# Error handling tests
test le-5.1 {Error handling - invalid tensor1} -body {
    ;# Create valid tensor2
    set tensor2 [create_tensor {1.0 2.0} {2}]
    
    ;# Test with non-existent tensor1
    if {[catch {torch::le "invalid_tensor" $tensor2} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test le-5.2 {Error handling - invalid tensor2} -body {
    ;# Create valid tensor1
    set tensor1 [create_tensor {1.0 2.0} {2}]
    
    ;# Test with non-existent tensor2
    if {[catch {torch::le $tensor1 "invalid_tensor"} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test le-5.3 {Error handling - missing required parameters} -body {
    ;# Test named syntax without required parameters
    if {[catch {torch::le} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test le-5.4 {Error handling - missing one tensor} -body {
    ;# Test positional syntax with missing tensor
    set tensor1 [create_tensor {1.0 2.0} {2}]
    if {[catch {torch::le $tensor1} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

test le-5.5 {Error handling - unknown parameter} -body {
    ;# Test named syntax with unknown parameter
    set tensor1 [create_tensor {1.0 2.0} {2}]
    set tensor2 [create_tensor {2.0 3.0} {2}]
    if {[catch {torch::le -unknown_param $tensor1 -input2 $tensor2} error]} {
        return "success"
    } else {
        return "failed: should have thrown error"
    }
} -result "success"

;# Test with different tensor shapes and dtypes
test le-6.1 {Different tensor dtypes - int32} -body {
    ;# Create integer tensors
    set tensor1 [create_tensor {1 3 5} {3} int32]
    set tensor2 [create_tensor {2 3 4} {3} int32]
    
    ;# Test comparison
    set result [torch::le -input1 $tensor1 -input2 $tensor2]
    
    ;# Should return boolean tensor with shape [3]
    set shape [torch::tensor_shape $result]
    
    if {$shape eq "3"} {
        return "success"
    } else {
        return "failed: expected shape 3, got $shape"
    }
} -result "success"

test le-6.2 {Different tensor shapes - 2D tensors} -body {
    ;# Create 2D tensors
    set tensor1 [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set tensor2 [create_tensor {1.5 1.5 3.0 5.0} {2 2}]
    
    ;# Test comparison
    set result [torch::le -input1 $tensor1 -input2 $tensor2]
    
    ;# Should return 2D boolean tensor with shape [2, 2]
    set shape [torch::tensor_shape $result]
    
    if {$shape eq "2 2"} {
        return "success"
    } else {
        return "failed: expected shape '2 2', got $shape"
    }
} -result "success"

;# Test parameter order flexibility
test le-7.1 {Parameter order flexibility} -body {
    ;# Create test tensors
    set tensor1 [create_tensor {1.0 2.0} {2}]
    set tensor2 [create_tensor {1.0 3.0} {2}]
    
    ;# Test different parameter orders
    set result1 [torch::le -input1 $tensor1 -input2 $tensor2]
    set result2 [torch::le -input2 $tensor2 -input1 $tensor1]
    
    ;# Both should work and return same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    if {$shape1 eq $shape2} {
        return "success"
    } else {
        return "failed: parameter order should be flexible"
    }
} -result "success"

;# Test scalar comparison
test le-8.1 {Scalar tensor comparison} -body {
    ;# Create scalar tensors
    set tensor1 [create_tensor {2.5} {1}]
    set tensor2 [create_tensor {3.0} {1}]
    
    ;# Test comparison
    set result [torch::le -input1 $tensor1 -input2 $tensor2]
    
    ;# Should return scalar boolean tensor with shape [1]
    set shape [torch::tensor_shape $result]
    
    if {$shape eq "1"} {
        return "success"
    } else {
        return "failed: expected shape 1, got $shape"
    }
} -result "success"

cleanupTests 