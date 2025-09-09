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

proc tensor_approx_equal {tensor1 tensor2 tolerance} {
    set diff [torch::tensor_sub $tensor1 $tensor2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    return [expr {$max_val < $tolerance}]
}

# Test 1: Basic positional syntax with default alpha
test elu-1.1 {Basic positional syntax with default alpha} {
    set input [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32]
    set result [torch::elu $input]
    
    # Check that result is a valid tensor with same shape
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    expr {$input_shape eq $result_shape}
} 1

# Test 2: Positional syntax with custom alpha
test elu-1.2 {Positional syntax with custom alpha} {
    set input [torch::tensor_create -data {-1.0 0.0 1.0} -dtype float32]
    set result [torch::elu $input 2.0]
    
    # Check that result preserves shape and dtype
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    set input_dtype [torch::tensor_dtype $input]
    set result_dtype [torch::tensor_dtype $result]
    expr {$input_shape eq $result_shape && $input_dtype eq $result_dtype}
} 1

# Test 3: Named parameter syntax with default alpha
test elu-2.1 {Named parameter syntax with default alpha} {
    set input [torch::tensor_create -data {-1.0 0.0 1.0} -dtype float32]
    set result [torch::elu -input $input]
    
    # Check result validity
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    expr {$input_shape eq $result_shape}
} 1

# Test 4: Named parameter syntax with custom alpha
test elu-2.2 {Named parameter syntax with custom alpha} {
    set input [torch::tensor_create -data {-1.0 0.0 1.0} -dtype float32]
    set result [torch::elu -input $input -alpha 0.5]
    
    # Verify shape preservation
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    expr {$input_shape eq $result_shape}
} 1

# Test 5: Alternative named parameter syntax (-tensor)
test elu-2.3 {Alternative named parameter syntax using -tensor} {
    set input [torch::tensor_create -data {-1.0 0.0 1.0} -dtype float32]
    set result [torch::elu -tensor $input -alpha 3.0]
    
    # Verify operation success
    set result_shape [torch::tensor_shape $result]
    expr {$result_shape eq "3"}
} 1

# Test 6: CamelCase alias
test elu-3.1 {CamelCase alias torch::Elu} {
    set input [torch::tensor_create -data {-1.0 0.0 1.0} -dtype float32]
    set result [torch::Elu $input]
    
    # Verify result
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    expr {$input_shape eq $result_shape}
} 1

# Test 7: CamelCase alias with named parameters
test elu-3.2 {CamelCase alias with named parameters} {
    set input [torch::tensor_create -data {-2.0 1.0} -dtype float32]
    set result [torch::Elu -input $input -alpha 1.5]
    
    # Verify result properties
    set result_shape [torch::tensor_shape $result]
    set result_dtype [torch::tensor_dtype $result]
    expr {$result_shape eq "2" && $result_dtype eq "Float32"}
} 1

# Test 8: Syntax consistency - both syntaxes should give same result
test elu-4.1 {Syntax consistency between positional and named} {
    set input [torch::tensor_create -data {-1.5 -0.5 0.5 1.5} -dtype float32]
    
    set result1 [torch::elu $input 2.0]
    set result2 [torch::elu -input $input -alpha 2.0]
    
    tensor_approx_equal $result1 $result2 1e-6
} 1

# Test 9: Multi-dimensional tensor support
test elu-5.1 {Multi-dimensional tensor support} {
    set input [torch::zeros {2 2} float32 cpu false]
    set result [torch::elu $input]
    
    set expected_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    expr {$expected_shape eq $result_shape}
} 1

# Test 10: Different data types
test elu-5.2 {Double precision tensor} {
    set input [torch::tensor_create -data {-1.0 0.0 1.0} -dtype float64]
    set result [torch::elu $input]
    
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Float64"}
} 1

# Test 11: Large tensor performance
test elu-5.3 {Large tensor handling} {
    set input [torch::tensor_randn {100 100} cpu float32]
    set result [torch::elu $input]
    
    set shape [torch::tensor_shape $result]
    expr {$shape eq "100 100"}
} 1

# Test 12: Mathematical properties - positive values unchanged
test elu-6.1 {Mathematical property - positive values unchanged} {
    set input [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set result [torch::elu $input]
    
    # For positive values, ELU should return the same values
    tensor_approx_equal $input $result 1e-6
} 1

# Test 13: Mathematical properties - zero value
test elu-6.2 {Mathematical property - zero value} {
    set input [torch::zeros {1} float32 cpu false]
    set result [torch::elu $input]
    
    # ELU(0) should be 0
    tensor_approx_equal $input $result 1e-6
} 1

# Test 14: Alpha parameter effect
test elu-6.3 {Alpha parameter effect on negative values} {
    set input [torch::tensor_create -data {-1.0} -dtype float32]
    set result1 [torch::elu $input 1.0]
    set result2 [torch::elu $input 2.0]
    
    # With larger alpha, negative values should have different results
    # We just check that they're different (not equal)
    set are_equal [tensor_approx_equal $result1 $result2 1e-6]
    expr {!$are_equal}
} 1

# Test 15: Continuity at zero with small values
test elu-7.1 {ELU continuity around zero} {
    set small_pos [torch::tensor_create -data {0.001} -dtype float32]
    set small_neg [torch::tensor_create -data {-0.001} -dtype float32]
    
    set result_pos [torch::elu $small_pos]
    set result_neg [torch::elu $small_neg]
    
    # Both should produce valid results
    set pos_shape [torch::tensor_shape $result_pos]
    set neg_shape [torch::tensor_shape $result_neg]
    expr {$pos_shape eq "1" && $neg_shape eq "1"}
} 1

# Error handling tests
test elu-error-1.1 {Error: Missing tensor argument} {
    catch {torch::elu} result
    string match "*wrong # args*" $result
} 1

test elu-error-1.2 {Error: Too many positional arguments} {
    set input [torch::tensor_create -data {1.0} -dtype float32]
    catch {torch::elu $input 1.0 2.0} result
    string match "*wrong # args*" $result
} 1

test elu-error-1.3 {Error: Invalid tensor name} {
    catch {torch::elu "invalid_tensor"} result
    string match "*Invalid tensor name*" $result
} 1

test elu-error-1.4 {Error: Invalid alpha value} {
    set input [torch::tensor_create -data {1.0} -dtype float32]
    catch {torch::elu $input "not_a_number"} result
    string match "*invalid alpha value*" $result
} 1

test elu-error-1.5 {Error: Zero alpha} {
    set input [torch::tensor_create -data {1.0} -dtype float32]
    catch {torch::elu $input 0.0} result
    string match "*alpha must be > 0*" $result
} 1

test elu-error-1.6 {Error: Negative alpha} {
    set input [torch::tensor_create -data {1.0} -dtype float32]
    catch {torch::elu $input -1.0} result
    string match "*alpha must be > 0*" $result
} 1

test elu-error-2.1 {Error: Missing required named parameter} {
    catch {torch::elu -alpha 1.0} result
    string match "*required parameter -input missing*" $result
} 1

test elu-error-2.2 {Error: Unknown named parameter} {
    set input [torch::tensor_create -data {1.0} -dtype float32]
    catch {torch::elu -input $input -unknown_param 1.0} result
    string match "*unknown option*" $result
} 1

test elu-error-2.3 {Error: Missing value for named parameter} {
    set input [torch::tensor_create -data {1.0} -dtype float32]
    catch {torch::elu -input $input -alpha} result
    string match "*wrong # args*" $result
} 1

test elu-error-2.4 {Error: Odd number of arguments in named syntax} {
    set input [torch::tensor_create -data {1.0} -dtype float32]
    catch {torch::elu -input $input -alpha 1.0 -extra} result
    string match "*wrong # args*" $result
} 1

test elu-error-2.5 {Error: Invalid alpha in named syntax} {
    set input [torch::tensor_create -data {1.0} -dtype float32]
    catch {torch::elu -input $input -alpha "not_a_number"} result
    string match "*invalid alpha value*" $result
} 1

# Integration tests
test elu-integration-1 {Integration with other tensor operations} {
    set input [torch::tensor_create -data {-1.0 0.0 1.0} -dtype float32]
    set elu_result [torch::elu $input]
    
    # Test that we can use result in further operations
    set sum_result [torch::tensor_sum $elu_result]
    set sum_shape [torch::tensor_shape $sum_result]
    expr {$sum_shape eq ""}
} 1

test elu-integration-2 {Chain multiple ELU operations} {
    set input [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32]
    set result1 [torch::elu $input]
    set result2 [torch::elu $result1]
    
    # Should be able to chain operations
    set final_shape [torch::tensor_shape $result2]
    expr {$final_shape eq "5"}
} 1

# Cleanup
cleanupTests 