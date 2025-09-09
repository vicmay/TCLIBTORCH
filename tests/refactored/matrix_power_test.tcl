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

# Test torch::matrix_power and torch::matrixPower
test matrix_power-1.1 {Basic matrix power - positional syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_power $A 2]
    # Verify it returns a valid tensor handle and has correct shape
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-1.2 {Basic matrix power - named syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_power -input $A -n 2]
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-1.3 {Basic matrix power - camelCase alias} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrixPower -input $A -n 2]
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-2.1 {Matrix power with n=0 - positional syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_power $A 0]
    # A^0 = I (identity matrix), verify shape and valid tensor
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-2.2 {Matrix power with n=0 - named syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_power -input $A -n 0]
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-2.3 {Matrix power with n=1 - positional syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_power $A 1]
    # A^1 = A, verify shape and valid tensor
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-2.4 {Matrix power with n=1 - named syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_power -input $A -n 1]
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-3.1 {Matrix power with n=3 - positional syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_power $A 3]
    # A^3, verify shape and valid tensor
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-3.2 {Matrix power with n=3 - named syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_power -input $A -n 3]
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-4.1 {Matrix power with negative n - positional syntax} {
    set A [torch::eye 2 2 float32 cpu false]
    set result [torch::matrix_power $A -1]
    # I^-1 = I (identity matrix), verify shape and valid tensor
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-4.2 {Matrix power with negative n - named syntax} {
    set A [torch::eye 2 2 float32 cpu false]
    set result [torch::matrix_power -input $A -n -1]
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-5.1 {Matrix power consistency between syntaxes} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result_pos [torch::matrix_power $A 2]
    set result_named [torch::matrix_power -input $A -n 2]
    set result_camel [torch::matrixPower -input $A -n 2]
    
    # Verify all three syntaxes return valid tensors with correct shape
    set shape_pos [torch::tensor_shape $result_pos]
    set shape_named [torch::tensor_shape $result_named]
    set shape_camel [torch::tensor_shape $result_camel]
    
    expr {[string match "tensor*" $result_pos] && [string match "tensor*" $result_named] && 
          [string match "tensor*" $result_camel] && $shape_pos eq "2 2" && 
          $shape_named eq "2 2" && $shape_camel eq "2 2"}
} {1}

test matrix_power-6.1 {Error handling - missing input (positional)} {
    catch {torch::matrix_power} msg
    string match "*input*" $msg
} {1}

test matrix_power-6.2 {Error handling - missing input (named)} {
    catch {torch::matrix_power -n 2} msg
    string match "*input*" $msg
} {1}

test matrix_power-6.3 {Error handling - missing n (positional)} {
    set A [torch::eye 2 2 float32 cpu false]
    catch {torch::matrix_power $A} msg
    string match "*input n*" $msg
} {1}

test matrix_power-6.4 {Error handling - invalid tensor name} {
    catch {torch::matrix_power invalid_tensor 2} msg
    string match "*Invalid input tensor*" $msg
} {1}

test matrix_power-6.5 {Error handling - invalid n parameter} {
    set A [torch::eye 2 2 float32 cpu false]
    catch {torch::matrix_power $A not_a_number} msg
    string match "*must be an integer*" $msg
} {1}

test matrix_power-6.6 {Error handling - unknown parameter} {
    set A [torch::eye 2 2 float32 cpu false]
    catch {torch::matrix_power -input $A -unknown_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test matrix_power-7.1 {Matrix power with identity matrix} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_power $A 5]
    # I^5 = I (identity matrix), verify shape and valid tensor
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "3 3"}
} {1}

test matrix_power-7.2 {Matrix power with 3x3 matrix} {
    set temp [torch::tensor_create {2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 2.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {3 3}]
    set result [torch::matrix_power $A 3]
    # (2I)^3 = 8I, verify shape and valid tensor
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "3 3"}
} {1}

test matrix_power-8.1 {Matrix power with batch of matrices} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2 2}]
    set result [torch::matrix_power $A 2]
    set shape [torch::tensor_shape $result]
    # Should preserve batch dimension
    set shape
} {2 2 2}

test matrix_power-8.2 {Matrix power with mixed syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_power $A -n 2]
    # Just verify that the mixed syntax works and produces a valid tensor
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-9.1 {Matrix power with large n} {
    set A [torch::eye 2 2 float32 cpu false]
    set result [torch::matrix_power $A 100]
    # I^100 = I, just verify shape and valid tensor
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

test matrix_power-9.2 {Matrix power with zeros matrix} {
    set A [torch::zeros {2 2} float32 cpu false]
    set result [torch::matrix_power $A 2]
    # 0^2 = 0, just verify shape and valid tensor
    set shape [torch::tensor_shape $result]
    expr {[string match "tensor*" $result] && $shape eq "2 2"}
} {1}

cleanupTests 