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

# Test torch::matrix_norm and torch::matrixNorm
test matrix_norm-1.1 {Basic matrix norm - positional syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_norm $A]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 5.477225575051661) < 1e-6}
} {1}

test matrix_norm-1.2 {Basic matrix norm - named syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_norm -input $A]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 5.477225575051661) < 1e-6}
} {1}

test matrix_norm-1.3 {Basic matrix norm - camelCase alias} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrixNorm -input $A]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 5.477225575051661) < 1e-6}
} {1}

test matrix_norm-2.1 {Matrix norm with frobenius norm - positional syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_norm $A fro]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 5.477225575051661) < 1e-6}
} {1}

test matrix_norm-2.2 {Matrix norm with frobenius norm - named syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_norm -input $A -ord fro]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 5.477225575051661) < 1e-6}
} {1}

test matrix_norm-2.3 {Matrix norm with nuclear norm - positional syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_norm $A nuc]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 5.830952) < 1e-6}
} {1}

test matrix_norm-2.4 {Matrix norm with nuclear norm - named syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_norm -input $A -ord nuc]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 5.830952) < 1e-6}
} {1}

test matrix_norm-3.1 {Matrix norm with numeric ord - positional syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_norm $A 2]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 5.46498570773804) < 1e-6}
} {1}

test matrix_norm-3.2 {Matrix norm with numeric ord - named syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_norm -input $A -ord 2]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 5.46498570773804) < 1e-6}
} {1}

test matrix_norm-4.1 {Matrix norm with specific dimensions - positional syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2 2}]
    set result [torch::matrix_norm $A fro {1 2}]
    set shape [torch::tensor_shape $result]
    set shape
} {2}

test matrix_norm-4.2 {Matrix norm with specific dimensions - named syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2 2}]
    set result [torch::matrix_norm -input $A -ord fro -dim {1 2}]
    set shape [torch::tensor_shape $result]
    set shape
} {2}

test matrix_norm-5.1 {Matrix norm with keepdim - positional syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2 2}]
    set result [torch::matrix_norm $A fro {1 2} 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 1 1}

test matrix_norm-5.2 {Matrix norm with keepdim - named syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2 2}]
    set result [torch::matrix_norm -input $A -ord fro -dim {1 2} -keepdim 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 1 1}

test matrix_norm-6.1 {Matrix norm consistency between syntaxes} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result_pos [torch::matrix_norm $A fro]
    set result_named [torch::matrix_norm -input $A -ord fro]
    set result_camel [torch::matrixNorm -input $A -ord fro]
    
    set val_pos [torch::tensor_item $result_pos]
    set val_named [torch::tensor_item $result_named]
    set val_camel [torch::tensor_item $result_camel]
    
    expr {abs($val_pos - $val_named) < 1e-10 && abs($val_named - $val_camel) < 1e-10}
} {1}

test matrix_norm-7.1 {Error handling - missing input (positional)} {
    catch {torch::matrix_norm} msg
    string match "*input*" $msg
} {1}

test matrix_norm-7.2 {Error handling - missing input (named)} {
    catch {torch::matrix_norm -ord fro} msg
    string match "*input*" $msg
} {1}

test matrix_norm-7.3 {Error handling - invalid tensor name} {
    catch {torch::matrix_norm invalid_tensor} msg
    string match "*Invalid input tensor*" $msg
} {1}

test matrix_norm-7.4 {Error handling - unknown parameter} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    catch {torch::matrix_norm -input $A -unknown_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test matrix_norm-8.1 {Matrix norm with 2x2 identity matrix} {
    set A [torch::eye 2 2 float32 cpu false]
    set result [torch::matrix_norm $A]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 1.4142135623730951) < 1e-6}
} {1}

test matrix_norm-8.2 {Matrix norm with 3D tensor} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2 2}]
    set result [torch::matrix_norm $A]
    set numel [torch::tensor_numel $result]
    expr {$numel > 0}
} {1}

test matrix_norm-9.1 {Matrix norm with integer tensor should fail} {
    set temp [torch::tensor_create {1 2 3 4} int32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    catch {torch::matrix_norm $A} msg
    string match "*Expected a floating point*" $msg
} {1}

test matrix_norm-9.2 {Matrix norm with zeros} {
    set A [torch::zeros {2 2} float32 cpu false]
    set result [torch::matrix_norm $A]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 0.0) < 1e-10}
} {1}

test matrix_norm-9.3 {Matrix norm with identity matrix} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_norm $A]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 1.7320508075688772) < 1e-6}
} {1}

test matrix_norm-10.1 {Matrix norm with mixed syntax} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_norm $A -ord fro]
    set norm_val [torch::tensor_item $result]
    expr {abs($norm_val - 5.477225575051661) < 1e-6}
} {1}

test matrix_norm-10.2 {Matrix norm with integer ord} {
    set temp [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set A [torch::tensor_reshape $temp {2 2}]
    set result [torch::matrix_norm $A 1]
    set norm_val [torch::tensor_item $result]
    expr {$norm_val > 0}
} {1}

cleanupTests 