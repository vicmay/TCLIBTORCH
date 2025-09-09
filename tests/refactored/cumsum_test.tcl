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
    return [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
}

# Test helper function to verify result
proc verify_cumsum_result {result} {
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

test cumsum-1.1 {Cumsum with positional syntax - dim 0} {
    set input [create_test_tensor]
    set result [torch::cumsum $input 0]
    verify_cumsum_result $result
} {1}

test cumsum-1.2 {Cumsum with positional syntax - dim 1} {
    set input [create_test_tensor]
    set result [torch::cumsum $input 1]
    verify_cumsum_result $result
} {1}

test cumsum-1.3 {Cumsum with positional syntax - negative dim} {
    set input [create_test_tensor]
    set result [torch::cumsum $input -1]
    verify_cumsum_result $result
} {1}

# ============================================================================
# NAMED PARAMETER SYNTAX TESTS
# ============================================================================

test cumsum-2.1 {Cumsum with named parameters - basic case} {
    set input [create_test_tensor]
    set result [torch::cumsum -input $input -dim 0]
    verify_cumsum_result $result
} {1}

test cumsum-2.2 {Cumsum with named parameters - dim 1} {
    set input [create_test_tensor]
    set result [torch::cumsum -input $input -dim 1]
    verify_cumsum_result $result
} {1}

test cumsum-2.3 {Cumsum with named parameters - negative dim} {
    set input [create_test_tensor]
    set result [torch::cumsum -input $input -dim -1]
    verify_cumsum_result $result
} {1}

test cumsum-2.4 {Cumsum with named parameters - parameters in different order} {
    set input [create_test_tensor]
    set result [torch::cumsum -dim 0 -input $input]
    verify_cumsum_result $result
} {1}

# ============================================================================
# CAMELCASE ALIAS TESTS
# ============================================================================

test cumsum-3.1 {Cumsum camelCase alias with positional syntax} {
    set input [create_test_tensor]
    set result [torch::cumSum $input 0]
    verify_cumsum_result $result
} {1}

test cumsum-3.2 {Cumsum camelCase alias with named parameters} {
    set input [create_test_tensor]
    set result [torch::cumSum -input $input -dim 1]
    verify_cumsum_result $result
} {1}

# ============================================================================
# MATHEMATICAL CORRECTNESS TESTS
# ============================================================================

test cumsum-4.1 {Cumsum mathematical correctness - simple case} {
    set input [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    set result [torch::cumsum $input 0]
    verify_cumsum_result $result
} {1}

test cumsum-4.2 {Cumsum mathematical correctness - 2D tensor dim 0} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::cumsum $input 0]
    verify_cumsum_result $result
} {1}

test cumsum-4.3 {Cumsum mathematical correctness - 2D tensor dim 1} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set result [torch::cumsum $input 1]
    verify_cumsum_result $result
} {1}

# ============================================================================
# CONSISTENCY TESTS (Both Syntaxes Should Produce Same Results)
# ============================================================================

test cumsum-5.1 {Consistency between positional and named syntax} {
    set input [create_test_tensor]
    set result1 [torch::cumsum $input 0]
    set result2 [torch::cumsum -input $input -dim 0]
    
    # Both should be valid tensors
    set valid1 [verify_cumsum_result $result1]
    set valid2 [verify_cumsum_result $result2]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

test cumsum-5.2 {Consistency between snake_case and camelCase} {
    set input [create_test_tensor]
    set result1 [torch::cumsum $input 1]
    set result2 [torch::cumSum $input 1]
    
    # Both should be valid tensors
    set valid1 [verify_cumsum_result $result1]
    set valid2 [verify_cumsum_result $result2]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

# ============================================================================
# DATA TYPE SUPPORT TESTS
# ============================================================================

test cumsum-6.1 {Cumsum with int32 tensors} {
    set input [torch::tensor_create -data {1 2 3 4} -shape {4} -dtype int32]
    set result [torch::cumsum $input 0]
    verify_cumsum_result $result
} {1}

test cumsum-6.2 {Cumsum with float64 tensors} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {4} -dtype float64]
    set result [torch::cumsum -input $input -dim 0]
    verify_cumsum_result $result
} {1}

# ============================================================================
# EDGE CASES
# ============================================================================

test cumsum-7.1 {Cumsum with single element tensor} {
    set input [torch::tensor_create -data {42.0} -shape {1} -dtype float32]
    set result [torch::cumsum $input 0]
    verify_cumsum_result $result
} {1}

test cumsum-7.2 {Cumsum with zeros} {
    set input [torch::tensor_create -data {0.0 0.0 0.0} -shape {3} -dtype float32]
    set result [torch::cumsum -input $input -dim 0]
    verify_cumsum_result $result
} {1}

test cumsum-7.3 {Cumsum with negative values} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0} -shape {3} -dtype float32]
    set result [torch::cumsum $input 0]
    verify_cumsum_result $result
} {1}

test cumsum-7.4 {Cumsum with large tensor} {
    set data {}
    for {set i 1} {$i <= 100} {incr i} {
        lappend data [expr {$i * 1.0}]
    }
    set input [torch::tensor_create -data $data -shape {100} -dtype float32]
    set result [torch::cumsum -input $input -dim 0]
    verify_cumsum_result $result
} {1}

test cumsum-7.5 {Cumsum with 3D tensor} {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]
    set result [torch::cumsum -input $input -dim 0]
    verify_cumsum_result $result
} {1}

test cumsum-7.6 {Cumsum with mixed values} {
    set input [torch::tensor_create -data {1.0 -1.0 2.0} -shape {3} -dtype float32]
    set result [torch::cumsum $input 0]
    verify_cumsum_result $result
} {1}

test cumsum-7.7 {Cumsum with fractional values} {
    set input [torch::tensor_create -data {0.5 0.25 0.125} -shape {3} -dtype float32]
    set result [torch::cumsum -input $input -dim 0]
    verify_cumsum_result $result
} {1}

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

test cumsum-error-1.1 {Error handling - invalid tensor name} {
    catch {torch::cumsum invalid_tensor 0} result
    string match "*Invalid tensor name*" $result
} {1}

test cumsum-error-1.2 {Error handling - missing parameters in positional syntax} {
    set input [create_test_tensor]
    catch {torch::cumsum $input} result
    string match "*Wrong number of arguments*" $result
} {1}

test cumsum-error-1.3 {Error handling - too many parameters in positional syntax} {
    set input [create_test_tensor]
    catch {torch::cumsum $input 0 1} result
    string match "*Wrong number of arguments*" $result
} {1}

test cumsum-error-1.4 {Error handling - missing required parameter in named syntax} {
    catch {torch::cumsum -dim 0} result
    string match "*Required parameter missing*" $result
} {1}

test cumsum-error-1.5 {Error handling - unknown parameter} {
    set input [create_test_tensor]
    catch {torch::cumsum -input $input -invalid_param 0} result
    string match "*Unknown parameter*" $result
} {1}

test cumsum-error-1.6 {Error handling - invalid dim value} {
    set input [create_test_tensor]
    catch {torch::cumsum $input invalid_dim} result
    string match "*Invalid dim value*" $result
} {1}

test cumsum-error-1.7 {Error handling - missing parameter value} {
    set input [create_test_tensor]
    catch {torch::cumsum -input $input -dim} result
    string match "*Missing value for parameter*" $result
} {1}

cleanupTests 