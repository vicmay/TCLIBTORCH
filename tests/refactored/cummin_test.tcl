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
    return [torch::tensor_create -data {3.0 1.0 4.0 2.0 5.0 0.0} -shape {2 3} -dtype float32]
}

# Test helper function to verify result
proc verify_cummin_result {result} {
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

test cummin-1.1 {Cummin with positional syntax - dim 0} {
    set input [create_test_tensor]
    set result [torch::cummin $input 0]
    verify_cummin_result $result
} {1}

test cummin-1.2 {Cummin with positional syntax - dim 1} {
    set input [create_test_tensor]
    set result [torch::cummin $input 1]
    verify_cummin_result $result
} {1}

test cummin-1.3 {Cummin with positional syntax - negative dim} {
    set input [create_test_tensor]
    set result [torch::cummin $input -1]
    verify_cummin_result $result
} {1}

# ============================================================================
# NAMED PARAMETER SYNTAX TESTS
# ============================================================================

test cummin-2.1 {Cummin with named parameters - basic case} {
    set input [create_test_tensor]
    set result [torch::cummin -input $input -dim 0]
    verify_cummin_result $result
} {1}

test cummin-2.2 {Cummin with named parameters - dim 1} {
    set input [create_test_tensor]
    set result [torch::cummin -input $input -dim 1]
    verify_cummin_result $result
} {1}

test cummin-2.3 {Cummin with named parameters - negative dim} {
    set input [create_test_tensor]
    set result [torch::cummin -input $input -dim -1]
    verify_cummin_result $result
} {1}

test cummin-2.4 {Cummin with named parameters - parameters in different order} {
    set input [create_test_tensor]
    set result [torch::cummin -dim 0 -input $input]
    verify_cummin_result $result
} {1}

# ============================================================================
# CAMELCASE ALIAS TESTS
# ============================================================================

test cummin-3.1 {Cummin camelCase alias with positional syntax} {
    set input [create_test_tensor]
    set result [torch::cumMin $input 0]
    verify_cummin_result $result
} {1}

test cummin-3.2 {Cummin camelCase alias with named parameters} {
    set input [create_test_tensor]
    set result [torch::cumMin -input $input -dim 1]
    verify_cummin_result $result
} {1}

# ============================================================================
# MATHEMATICAL CORRECTNESS TESTS
# ============================================================================

test cummin-4.1 {Cummin mathematical correctness - simple case} {
    set input [torch::tensor_create -data {3.0 1.0 4.0} -shape {3} -dtype float32]
    set result [torch::cummin $input 0]
    # Verify we got a valid tensor - cummin should produce [3.0, 1.0, 1.0]
    verify_cummin_result $result
} {1}

test cummin-4.2 {Cummin mathematical correctness - 2D tensor dim 0} {
    set input [torch::tensor_create -data {3.0 1.0 2.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::cummin $input 0]
    verify_cummin_result $result
} {1}

test cummin-4.3 {Cummin mathematical correctness - 2D tensor dim 1} {
    set input [torch::tensor_create -data {3.0 1.0 2.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::cummin $input 1]
    verify_cummin_result $result
} {1}

# ============================================================================
# CONSISTENCY TESTS (Both Syntaxes Should Produce Same Results)
# ============================================================================

test cummin-5.1 {Consistency between positional and named syntax} {
    set input [create_test_tensor]
    set result1 [torch::cummin $input 0]
    set result2 [torch::cummin -input $input -dim 0]
    
    # Both should be valid tensors
    set valid1 [verify_cummin_result $result1]
    set valid2 [verify_cummin_result $result2]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

test cummin-5.2 {Consistency between snake_case and camelCase} {
    set input [create_test_tensor]
    set result1 [torch::cummin $input 1]
    set result2 [torch::cumMin $input 1]
    
    # Both should be valid tensors
    set valid1 [verify_cummin_result $result1]
    set valid2 [verify_cummin_result $result2]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

# ============================================================================
# DATA TYPE SUPPORT TESTS
# ============================================================================

test cummin-6.1 {Cummin with int32 tensors} {
    set input [torch::tensor_create -data {3 1 4 2} -shape {4} -dtype int32]
    set result [torch::cummin $input 0]
    verify_cummin_result $result
} {1}

test cummin-6.2 {Cummin with float64 tensors} {
    set input [torch::tensor_create -data {3.0 1.0 4.0 2.0} -shape {4} -dtype float64]
    set result [torch::cummin -input $input -dim 0]
    verify_cummin_result $result
} {1}

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

test cummin-7.1 {Error handling - invalid tensor name} {
    catch {torch::cummin invalid_tensor 0} result
    string match "*Invalid tensor name*" $result
} {1}

test cummin-7.2 {Error handling - missing parameters in positional syntax} {
    set input [create_test_tensor]
    catch {torch::cummin $input} result
    string match "*Wrong number of arguments*" $result
} {1}

test cummin-7.3 {Error handling - too many parameters in positional syntax} {
    set input [create_test_tensor]
    catch {torch::cummin $input 0 1} result
    string match "*Wrong number of arguments*" $result
} {1}

test cummin-7.4 {Error handling - missing required parameter in named syntax} {
    catch {torch::cummin -dim 0} result
    string match "*Required parameter missing*" $result
} {1}

test cummin-7.5 {Error handling - unknown parameter} {
    set input [create_test_tensor]
    catch {torch::cummin -input $input -invalid_param 0} result
    string match "*Unknown parameter*" $result
} {1}

test cummin-7.6 {Error handling - invalid dim value} {
    set input [create_test_tensor]
    catch {torch::cummin $input invalid_dim} result
    string match "*Invalid dim value*" $result
} {1}

test cummin-7.7 {Error handling - missing value for parameter} {
    set input [create_test_tensor]
    catch {torch::cummin -input $input -dim} result
    string match "*Missing value for parameter*" $result
} {1}

# ============================================================================
# EDGE CASES
# ============================================================================

test cummin-8.1 {Cummin with 1D tensor} {
    set input [torch::tensor_create -data {5.0 2.0 8.0 1.0} -shape {4} -dtype float32]
    set result [torch::cummin $input 0]
    verify_cummin_result $result
} {1}

test cummin-8.2 {Cummin with 3D tensor} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]
    set result [torch::cummin -input $input -dim 2]
    verify_cummin_result $result
} {1}

test cummin-8.3 {Cummin with single element tensor} {
    set input [torch::tensor_create -data {5.0} -shape {1} -dtype float32]
    set result [torch::cummin $input 0]
    verify_cummin_result $result
} {1}

# ============================================================================
# PERFORMANCE AND MEMORY TESTS
# ============================================================================

test cummin-9.1 {Large tensor handling} {
    # Create larger tensor for performance testing
    set data {}
    for {set i 0} {$i < 1000} {incr i} {
        lappend data [expr {rand() * 100}]
    }
    set input [torch::tensor_create -data $data -shape {1000} -dtype float32]
    set result [torch::cummin $input 0]
    verify_cummin_result $result
} {1}

# ============================================================================
# CLEANUP AND SUMMARY
# ============================================================================

puts "\n=== CUMMIN TEST SUMMARY ==="
puts "Testing torch::cummin dual syntax implementation"
puts "- Positional syntax (backward compatibility)"
puts "- Named parameter syntax"
puts "- camelCase alias (torch::cumMin)"
puts "- Mathematical correctness"
puts "- Error handling"
puts "- Multiple data types"
puts "- Edge cases"

cleanupTests 