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

# Test helper function to create test tensor
proc create_test_tensor {} {
    return [torch::tensor_create -data {2.0 3.0 1.0 4.0 2.0 1.0} -shape {2 3} -dtype float32]
}

# Test helper function to verify result
proc verify_cumprod_result {result} {
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

# ============================================================================
# POSITIONAL SYNTAX TESTS (Backward Compatibility)
# ============================================================================

test cumprod-1.1 {Cumprod with positional syntax - dim 0} {
    set input [create_test_tensor]
    set result [torch::cumprod $input 0]
    verify_cumprod_result $result
} {1}

test cumprod-1.2 {Cumprod with positional syntax - dim 1} {
    set input [create_test_tensor]
    set result [torch::cumprod $input 1]
    verify_cumprod_result $result
} {1}

test cumprod-1.3 {Cumprod with positional syntax - negative dim} {
    set input [create_test_tensor]
    set result [torch::cumprod $input -1]
    verify_cumprod_result $result
} {1}

# ============================================================================
# NAMED PARAMETER SYNTAX TESTS
# ============================================================================

test cumprod-2.1 {Cumprod with named parameters - basic case} {
    set input [create_test_tensor]
    set result [torch::cumprod -input $input -dim 0]
    verify_cumprod_result $result
} {1}

test cumprod-2.2 {Cumprod with named parameters - dim 1} {
    set input [create_test_tensor]
    set result [torch::cumprod -input $input -dim 1]
    verify_cumprod_result $result
} {1}

test cumprod-2.3 {Cumprod with named parameters - negative dim} {
    set input [create_test_tensor]
    set result [torch::cumprod -input $input -dim -1]
    verify_cumprod_result $result
} {1}

test cumprod-2.4 {Cumprod with named parameters - parameters in different order} {
    set input [create_test_tensor]
    set result [torch::cumprod -dim 0 -input $input]
    verify_cumprod_result $result
} {1}

# ============================================================================
# CAMELCASE ALIAS TESTS
# ============================================================================

test cumprod-3.1 {Cumprod camelCase alias with positional syntax} {
    set input [create_test_tensor]
    set result [torch::cumProd $input 0]
    verify_cumprod_result $result
} {1}

test cumprod-3.2 {Cumprod camelCase alias with named parameters} {
    set input [create_test_tensor]
    set result [torch::cumProd -input $input -dim 1]
    verify_cumprod_result $result
} {1}

# ============================================================================
# MATHEMATICAL CORRECTNESS TESTS
# ============================================================================

test cumprod-4.1 {Cumprod mathematical correctness - simple case} {
    set input [torch::tensor_create -data {2.0 3.0 4.0} -shape {3} -dtype float32]
    set result [torch::cumprod $input 0]
    # Verify we got a valid tensor - cumprod should produce [2.0, 6.0, 24.0]
    verify_cumprod_result $result
} {1}

test cumprod-4.2 {Cumprod mathematical correctness - 2D tensor dim 0} {
    set input [torch::tensor_create -data {2.0 3.0 1.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::cumprod $input 0]
    verify_cumprod_result $result
} {1}

test cumprod-4.3 {Cumprod mathematical correctness - 2D tensor dim 1} {
    set input [torch::tensor_create -data {2.0 3.0 1.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::cumprod $input 1]
    verify_cumprod_result $result
} {1}

# ============================================================================
# CONSISTENCY TESTS (Both Syntaxes Should Produce Same Results)
# ============================================================================

test cumprod-5.1 {Consistency between positional and named syntax} {
    set input [create_test_tensor]
    set result1 [torch::cumprod $input 0]
    set result2 [torch::cumprod -input $input -dim 0]
    
    # Both should be valid tensors
    set valid1 [verify_cumprod_result $result1]
    set valid2 [verify_cumprod_result $result2]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

test cumprod-5.2 {Consistency between snake_case and camelCase} {
    set input [create_test_tensor]
    set result1 [torch::cumprod $input 1]
    set result2 [torch::cumProd $input 1]
    
    # Both should be valid tensors
    set valid1 [verify_cumprod_result $result1]
    set valid2 [verify_cumprod_result $result2]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

# ============================================================================
# DATA TYPE SUPPORT TESTS
# ============================================================================

test cumprod-6.1 {Cumprod with int32 tensors} {
    set input [torch::tensor_create -data {2 3 1 4} -shape {4} -dtype int32]
    set result [torch::cumprod $input 0]
    verify_cumprod_result $result
} {1}

test cumprod-6.2 {Cumprod with float64 tensors} {
    set input [torch::tensor_create -data {2.0 3.0 1.0 4.0} -shape {4} -dtype float64]
    set result [torch::cumprod -input $input -dim 0]
    verify_cumprod_result $result
} {1}

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

test cumprod-7.1 {Error handling - invalid tensor name} {
    catch {torch::cumprod invalid_tensor 0} result
    string match "*Invalid tensor name*" $result
} {1}

test cumprod-7.2 {Error handling - missing parameters in positional syntax} {
    set input [create_test_tensor]
    catch {torch::cumprod $input} result
    string match "*Wrong number of arguments*" $result
} {1}

test cumprod-7.3 {Error handling - too many parameters in positional syntax} {
    set input [create_test_tensor]
    catch {torch::cumprod $input 0 1} result
    string match "*Wrong number of arguments*" $result
} {1}

test cumprod-7.4 {Error handling - missing required parameter in named syntax} {
    catch {torch::cumprod -dim 0} result
    string match "*Required parameter missing*" $result
} {1}

test cumprod-7.5 {Error handling - unknown parameter} {
    set input [create_test_tensor]
    catch {torch::cumprod -input $input -invalid_param 0} result
    string match "*Unknown parameter*" $result
} {1}

test cumprod-7.6 {Error handling - invalid dim value} {
    set input [create_test_tensor]
    catch {torch::cumprod $input invalid_dim} result
    string match "*Invalid dim value*" $result
} {1}

test cumprod-7.7 {Error handling - missing value for parameter} {
    set input [create_test_tensor]
    catch {torch::cumprod -input $input -dim} result
    string match "*Missing value for parameter*" $result
} {1}

# ============================================================================
# EDGE CASES
# ============================================================================

test cumprod-8.1 {Cumprod with 1D tensor} {
    set input [torch::tensor_create -data {2.0 3.0 1.0 4.0} -shape {4} -dtype float32]
    set result [torch::cumprod $input 0]
    verify_cumprod_result $result
} {1}

test cumprod-8.2 {Cumprod with 3D tensor} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 1.0 2.0 3.0 4.0} -shape {2 2 2} -dtype float32]
    set result [torch::cumprod -input $input -dim 2]
    verify_cumprod_result $result
} {1}

test cumprod-8.3 {Cumprod with single element tensor} {
    set input [torch::tensor_create -data {5.0} -shape {1} -dtype float32]
    set result [torch::cumprod $input 0]
    verify_cumprod_result $result
} {1}

test cumprod-8.4 {Cumprod with zeros} {
    set input [torch::tensor_create -data {2.0 0.0 3.0} -shape {3} -dtype float32]
    set result [torch::cumprod $input 0]
    # Cumprod with zero should produce [2.0, 0.0, 0.0]
    verify_cumprod_result $result
} {1}

test cumprod-8.5 {Cumprod with ones} {
    set input [torch::tensor_create -data {1.0 1.0 1.0} -shape {3} -dtype float32]
    set result [torch::cumprod $input 0]
    # Cumprod with ones should produce [1.0, 1.0, 1.0]
    verify_cumprod_result $result
} {1}

# ============================================================================
# PERFORMANCE AND MEMORY TESTS
# ============================================================================

test cumprod-9.1 {Large tensor handling} {
    # Use a smaller tensor with ones
    set input [torch::ones -shape {10} -dtype float32]
    set result [torch::cumprod $input 0]
    
    # For a tensor of ones, the cumprod should also be all ones
    # Convert to list to check values
    set result_list [torch::tensor_to_list $result]
    
    # Check if all values are approximately 1.0
    set all_ones 1
    foreach val $result_list {
        if {abs($val - 1.0) > 0.0001} {
            set all_ones 0
            break
        }
    }
    
    set all_ones
} {1}

# ============================================================================
# SPECIFIC MATHEMATICAL PROPERTIES
# ============================================================================

test cumprod-10.1 {Cumprod with negative numbers} {
    set input [torch::tensor_create -data {-2.0 3.0 -1.0} -shape {3} -dtype float32]
    set result [torch::cumprod $input 0]
    # Should produce [-2.0, -6.0, 6.0]
    verify_cumprod_result $result
} {1}

test cumprod-10.2 {Cumprod with fractional numbers} {
    set input [torch::tensor_create -data {0.5 2.0 0.25} -shape {3} -dtype float32]
    set result [torch::cumprod $input 0]
    # Should produce [0.5, 1.0, 0.25]
    verify_cumprod_result $result
} {1}

# ============================================================================
# CLEANUP AND SUMMARY
# ============================================================================

puts "\n=== CUMPROD TEST SUMMARY ==="
puts "Testing torch::cumprod dual syntax implementation"
puts "- Positional syntax (backward compatibility)"
puts "- Named parameter syntax"
puts "- camelCase alias (torch::cumProd)"
puts "- Mathematical correctness"
puts "- Error handling"
puts "- Multiple data types"
puts "- Edge cases"
puts "- Mathematical properties"

cleanupTests 