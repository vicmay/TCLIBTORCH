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

# Test torch::matrix_rank and torch::matrixRank
test matrix_rank-1.1 {Basic matrix rank - positional syntax} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank $A]
    # Verify it returns a valid tensor handle
    string match "tensor*" $result
} {1}

test matrix_rank-1.2 {Basic matrix rank - named syntax} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank -input $A]
    string match "tensor*" $result
} {1}

test matrix_rank-1.3 {Basic matrix rank - camelCase alias} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrixRank -input $A]
    string match "tensor*" $result
} {1}

test matrix_rank-2.1 {Matrix rank with tolerance - positional syntax} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank $A 1e-6]
    string match "tensor*" $result
} {1}

test matrix_rank-2.2 {Matrix rank with tolerance - named syntax} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank -input $A -tol 1e-6]
    string match "tensor*" $result
} {1}

test matrix_rank-2.3 {Matrix rank with tolerance - mixed syntax} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank $A -tol 1e-6]
    string match "tensor*" $result
} {1}

test matrix_rank-2.4 {Matrix rank with tolerance alias} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank -input $A -tolerance 1e-6]
    string match "tensor*" $result
} {1}

test matrix_rank-3.1 {Matrix rank with hermitian flag - positional syntax} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank $A 1e-12 1]
    string match "tensor*" $result
} {1}

test matrix_rank-3.2 {Matrix rank with hermitian flag - named syntax} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank -input $A -hermitian 1]
    string match "tensor*" $result
} {1}

test matrix_rank-3.3 {Matrix rank with hermitian flag - mixed syntax} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank $A -hermitian 1]
    string match "tensor*" $result
} {1}

test matrix_rank-3.4 {Matrix rank with hermitian false} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank -input $A -hermitian 0]
    string match "tensor*" $result
} {1}

test matrix_rank-4.1 {Matrix rank with both tolerance and hermitian - positional} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank $A 1e-6 1]
    string match "tensor*" $result
} {1}

test matrix_rank-4.2 {Matrix rank with both tolerance and hermitian - named} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank -input $A -tol 1e-6 -hermitian 1]
    string match "tensor*" $result
} {1}

test matrix_rank-4.3 {Matrix rank with both tolerance and hermitian - mixed} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank $A -tol 1e-6 -hermitian 1]
    string match "tensor*" $result
} {1}

test matrix_rank-5.1 {Matrix rank consistency between syntaxes} {
    set A [torch::eye 3 3 float32 cpu false]
    set result_pos [torch::matrix_rank $A]
    set result_named [torch::matrix_rank -input $A]
    set result_camel [torch::matrixRank -input $A]
    
    # Verify all three syntaxes return valid tensors
    expr {[string match "tensor*" $result_pos] && [string match "tensor*" $result_named] && [string match "tensor*" $result_camel]}
} {1}

test matrix_rank-6.1 {Matrix rank with singular matrix} {
    # Create a singular matrix (rank-deficient)
    set data [torch::tensor_create {1.0 2.0 3.0 2.0 4.0 6.0 3.0 6.0 9.0} float32 cpu false]
    set A [torch::tensor_reshape $data {3 3}]
    set result [torch::matrix_rank $A]
    string match "tensor*" $result
} {1}

test matrix_rank-6.2 {Matrix rank with zeros matrix} {
    set A [torch::zeros {3 3} float32 cpu false]
    set result [torch::matrix_rank $A]
    string match "tensor*" $result
} {1}

test matrix_rank-6.3 {Matrix rank with rectangular matrix} {
    set A [torch::ones {3 2} float32 cpu false]
    set result [torch::matrix_rank $A]
    string match "tensor*" $result
} {1}

test matrix_rank-7.1 {Error handling - missing input (positional)} {
    catch {torch::matrix_rank} msg
    string match "*input*" $msg
} {1}

test matrix_rank-7.2 {Error handling - missing input (named)} {
    catch {torch::matrix_rank -tol 1e-6} msg
    string match "*input*" $msg
} {1}

test matrix_rank-7.3 {Error handling - invalid tensor name} {
    catch {torch::matrix_rank invalid_tensor} msg
    string match "*Invalid input tensor*" $msg
} {1}

test matrix_rank-7.4 {Error handling - invalid tolerance} {
    set A [torch::eye 3 3 float32 cpu false]
    catch {torch::matrix_rank $A not_a_number} msg
    string match "*must be a number*" $msg
} {1}

test matrix_rank-7.5 {Error handling - invalid hermitian flag} {
    set A [torch::eye 3 3 float32 cpu false]
    catch {torch::matrix_rank $A 1e-12 not_a_bool} msg
    string match "*must be a boolean*" $msg
} {1}

test matrix_rank-7.6 {Error handling - unknown parameter} {
    set A [torch::eye 3 3 float32 cpu false]
    catch {torch::matrix_rank -input $A -unknown_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test matrix_rank-7.7 {Error handling - missing parameter value} {
    set A [torch::eye 3 3 float32 cpu false]
    catch {torch::matrix_rank -input $A -tol} msg
    string match "*Missing value*" $msg
} {1}

test matrix_rank-8.1 {Matrix rank with batch of matrices} {
    set data [torch::tensor_create {1.0 0.0 0.0 1.0 2.0 0.0 0.0 2.0} float32 cpu false]
    set A [torch::tensor_reshape $data {2 2 2}]
    set result [torch::matrix_rank $A]
    string match "tensor*" $result
} {1}

test matrix_rank-8.2 {Matrix rank with different dtypes} {
    set A [torch::eye 3 3 float64 cpu false]
    set result [torch::matrix_rank $A]
    string match "tensor*" $result
} {1}

test matrix_rank-9.1 {Matrix rank with very small tolerance} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank $A 1e-15]
    string match "tensor*" $result
} {1}

test matrix_rank-9.2 {Matrix rank with large tolerance} {
    set A [torch::eye 3 3 float32 cpu false]
    set result [torch::matrix_rank $A 1e-1]
    string match "tensor*" $result
} {1}

test matrix_rank-10.1 {Matrix rank parameter order independence} {
    set A [torch::eye 3 3 float32 cpu false]
    set result1 [torch::matrix_rank -input $A -tol 1e-6 -hermitian 1]
    set result2 [torch::matrix_rank -tol 1e-6 -input $A -hermitian 1]
    set result3 [torch::matrix_rank -hermitian 1 -input $A -tol 1e-6]
    
    # Verify all return valid tensors
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2] && [string match "tensor*" $result3]}
} {1}

cleanupTests 