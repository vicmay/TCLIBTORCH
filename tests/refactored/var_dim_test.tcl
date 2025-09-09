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

;# Test cases for positional syntax
test var-dim-1.1 {Basic positional syntax with default parameters} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 3}]
    
    ;# Test variance along dimension 0
    set result [torch::var_dim $input 0]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test var-dim-1.2 {Positional syntax with unbiased parameter} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 3}]
    
    ;# Test variance along dimension 1 with unbiased=false
    set result [torch::var_dim $input 1 0]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test var-dim-1.3 {Positional syntax with all parameters} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 3}]
    
    ;# Test variance along dimension 0 with unbiased=false and keepdim=true
    set result [torch::var_dim $input 0 0 1]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

;# Test cases for named parameter syntax
test var-dim-2.1 {Named parameter syntax with required parameters} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 3}]
    
    ;# Test variance along dimension 1
    set result [torch::var_dim -input $input -dim 1]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test var-dim-2.2 {Named parameter syntax with unbiased parameter} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 3}]
    
    ;# Test variance along dimension 0 with unbiased=false
    set result [torch::var_dim -input $input -dim 0 -unbiased 0]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test var-dim-2.3 {Named parameter syntax with keepdim parameter} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 3}]
    
    ;# Test variance along dimension 1 with keepdim=true
    set result [torch::var_dim -input $input -dim 1 -keepdim 1]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test var-dim-2.4 {Named parameter syntax with all parameters} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 3}]
    
    ;# Test variance along dimension 0 with all parameters
    set result [torch::var_dim -input $input -dim 0 -unbiased 0 -keepdim 1]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

;# Test cases for camelCase alias
test var-dim-3.1 {CamelCase alias - positional syntax} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 3}]
    
    ;# Test variance along dimension 0 using camelCase alias
    set result [torch::varDim $input 0]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test var-dim-3.2 {CamelCase alias - named parameter syntax} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 3}]
    
    ;# Test variance along dimension 1 using camelCase alias with named parameters
    set result [torch::varDim -input $input -dim 1 -unbiased 1 -keepdim 0]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

;# Error handling tests
test var-dim-4.1 {Error handling - invalid tensor} {
    ;# Test with non-existent tensor
    catch {torch::var_dim "nonexistent" 0} result
    expr {[string first "Invalid tensor name" $result] >= 0}
} {1}

test var-dim-4.2 {Error handling - missing dim parameter} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 2}]
    
    ;# Test without dim parameter
    catch {torch::var_dim $input} result
    expr {[string first "Invalid number of arguments" $result] >= 0}
} {1}

test var-dim-4.3 {Error handling - invalid dim value} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 2}]
    
    ;# Test with invalid dim value
    catch {torch::var_dim $input "invalid"} result
    expr {[string first "Invalid dim value" $result] >= 0}
} {1}

test var-dim-4.4 {Error handling - missing named parameters} {
    ;# Test with missing value for named parameter
    catch {torch::var_dim -input} result
    expr {[string first "Missing value for parameter" $result] >= 0}
} {1}

test var-dim-4.5 {Error handling - unknown named parameter} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 2}]
    
    ;# Test with unknown parameter
    catch {torch::var_dim -input $input -dim 0 -unknown 1} result
    expr {[string first "Unknown parameter" $result] >= 0}
} {1}

;# Mathematical correctness tests
test var-dim-5.1 {Mathematical correctness - simple variance} {
    ;# Create 1D tensor with known values
    set input [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu false]
    
    ;# Test variance along dimension 0
    set result [torch::var_dim $input 0]
    
    ;# Get the result value
    set result_value [torch::tensor_to_list $result]
    
    ;# Expected variance (unbiased): Σ(x-mean)²/(n-1) = 10/4 = 2.5
    ;# Check if result is close to expected value (allowing for floating point precision)
    set expected 2.5
    set tolerance 0.1
    expr {abs([lindex $result_value 0] - $expected) < $tolerance}
} {1}

test var-dim-5.2 {Mathematical correctness - unbiased vs biased} {
    ;# Create 1D tensor
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    
    ;# Test unbiased variance (default)
    set unbiased_result [torch::var_dim $input 0 1]
    set unbiased_value [torch::tensor_to_list $unbiased_result]
    
    ;# Test biased variance
    set biased_result [torch::var_dim $input 0 0]
    set biased_value [torch::tensor_to_list $biased_result]
    
    ;# Biased variance should be smaller than unbiased variance
    expr {[lindex $biased_value 0] < [lindex $unbiased_value 0]}
} {1}

test var-dim-5.3 {Mathematical correctness - keepdim parameter} {
    ;# Create 2D tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 2}]
    
    ;# Test variance without keepdim
    set result_no_keep [torch::var_dim $input 0 1 0]
    set shape_no_keep [torch::tensor_shape $result_no_keep]
    
    ;# Test variance with keepdim
    set result_keep [torch::var_dim $input 0 1 1]
    set shape_keep [torch::tensor_shape $result_keep]
    
    ;# With keepdim, the result should have more dimensions
    expr {[llength $shape_keep] > [llength $shape_no_keep]}
} {1}

;# Data type support tests
test var-dim-6.1 {Data type support - float32} {
    ;# Create 2D input tensor with float32
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 2}]
    
    ;# Test variance
    set result [torch::var_dim $input 0]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test var-dim-6.2 {Data type support - float64} {
    ;# Create 2D input tensor with float64
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float64 cpu false]
    set input [torch::tensor_reshape $input_1d {2 2}]
    
    ;# Test variance
    set result [torch::var_dim $input 0]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

;# Edge cases tests
test var-dim-7.1 {Edge case - single element tensor} {
    ;# Create single element tensor
    set input [torch::tensor_create {5.0} float32 cpu false]
    
    ;# Test variance along dimension 0
    set result [torch::var_dim $input 0]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test var-dim-7.2 {Edge case - all same values} {
    ;# Create tensor with all same values
    set input [torch::tensor_create {2.0 2.0 2.0 2.0} float32 cpu false]
    
    ;# Test variance along dimension 0
    set result [torch::var_dim $input 0]
    set result_value [torch::tensor_to_list $result]
    
    ;# Variance of same values should be 0 (or very close to 0)
    expr {abs([lindex $result_value 0]) < 1e-6}
} {1}

test var-dim-7.3 {Edge case - negative dimension} {
    ;# Create 2D tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 2}]
    
    ;# Test variance with negative dimension (-1 should be equivalent to 1)
    set result [torch::var_dim $input -1]
    
    ;# Check result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

;# Syntax consistency tests
test var-dim-8.1 {Syntax consistency - positional vs named} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 3}]
    
    ;# Test with positional syntax
    set result_pos [torch::var_dim $input 0 1 0]
    set value_pos [torch::tensor_to_list $result_pos]
    
    ;# Test with named syntax
    set result_named [torch::var_dim -input $input -dim 0 -unbiased 1 -keepdim 0]
    set value_named [torch::tensor_to_list $result_named]
    
    ;# Results should be identical
    expr {abs([lindex $value_pos 0] - [lindex $value_named 0]) < 1e-6}
} {1}

test var-dim-8.2 {Syntax consistency - snake_case vs camelCase} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 3}]
    
    ;# Test with snake_case
    set result_snake [torch::var_dim $input 1 1 0]
    set value_snake [torch::tensor_to_list $result_snake]
    
    ;# Test with camelCase
    set result_camel [torch::varDim $input 1 1 0]
    set value_camel [torch::tensor_to_list $result_camel]
    
    ;# Results should be identical
    expr {abs([lindex $value_snake 0] - [lindex $value_camel 0]) < 1e-6}
} {1}

test var-dim-8.3 {Syntax consistency - different parameter orders} {
    ;# Create 2D input tensor
    set input_1d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
    set input [torch::tensor_reshape $input_1d {2 3}]
    
    ;# Test with different parameter orders in named syntax
    set result1 [torch::var_dim -input $input -dim 0 -unbiased 1 -keepdim 0]
    set value1 [torch::tensor_to_list $result1]
    
    set result2 [torch::var_dim -dim 0 -input $input -keepdim 0 -unbiased 1]
    set value2 [torch::tensor_to_list $result2]
    
    ;# Results should be identical regardless of parameter order
    expr {abs([lindex $value1 0] - [lindex $value2 0]) < 1e-6}
} {1}

cleanupTests 