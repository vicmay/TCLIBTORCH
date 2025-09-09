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

# Helper function to create test tensors
proc createTestTensor {name values dtype device} {
    set tensor [torch::tensor_create -data $values -dtype $dtype -device $device -requiresGrad false]
    return $tensor
}

# Helper function to check if tensors are approximately equal
proc tensorsApproxEqual {tensor1 tensor2 tolerance} {
    # Check if tensors are boolean type
    set dtype1 [torch::tensor_dtype $tensor1]
    set dtype2 [torch::tensor_dtype $tensor2]
    
    if {$dtype1 eq "Bool" && $dtype2 eq "Bool"} {
        # For boolean tensors, use XOR to find differences, then sum to count them
        # If tensors are equal, XOR should be all zeros, sum should be 0
        set xor_result [torch::logical_xor $tensor1 $tensor2]
        set sum_diff [torch::tensor_sum $xor_result]
        set total_diff [torch::tensor_item $sum_diff]
        return [expr {$total_diff == 0}]
    } else {
        # For numeric tensors, use difference-based comparison
        set diff [torch::tensor_sub $tensor1 $tensor2]
        set abs_diff [torch::tensor_abs $diff]
        set max_diff [torch::tensor_max $abs_diff]
        set max_val [torch::tensor_item $max_diff]
        return [expr {$max_val < $tolerance}]
    }
}

# Helper function to create boolean tensors
proc createBoolTensor {name values device} {
    # Convert true/false to 1/0 for tensor creation
    set numeric_values {}
    foreach val $values {
        if {$val eq "true"} {
            lappend numeric_values 1
        } elseif {$val eq "false"} {
            lappend numeric_values 0
        } else {
            lappend numeric_values $val
        }
    }
    set tensor [torch::tensor_create -data $numeric_values -dtype bool -device $device -requiresGrad false]
    return $tensor
}

#===============================================================================
# POSITIONAL SYNTAX TESTS (Backward Compatibility)
#===============================================================================

test isclose-positional-1.1 {Basic isclose with positional syntax - identical tensors} {
    set input1 [createTestTensor "input1" {1.0 2.0 3.0 4.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.0 2.0 3.0 4.0} float32 cpu]
    set result [torch::isclose $input1 $input2]
    
    set expected [createBoolTensor "expected" {true true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-positional-1.2 {Isclose with positional syntax - small differences} {
    set input1 [createTestTensor "input1" {1.0 2.0 3.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.000001 2.000001 3.000001} float32 cpu]
    set result [torch::isclose $input1 $input2]
    
    ;# Small differences should be close with default tolerance
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-positional-1.3 {Isclose with positional syntax - large differences} {
    set input1 [createTestTensor "input1" {1.0 2.0 3.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.1 2.1 3.1} float32 cpu]
    set result [torch::isclose $input1 $input2]
    
    ;# Large differences should not be close with default tolerance
    set expected [createBoolTensor "expected" {false false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-positional-1.4 {Isclose with custom rtol - positional syntax} {
    set input1 [createTestTensor "input1" {1.0 2.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.05 2.05} float32 cpu]
    set result [torch::isclose $input1 $input2 0.1]
    
    ;# 5% difference should be close with rtol=0.1
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-positional-1.5 {Isclose with custom rtol and atol - positional syntax} {
    set input1 [createTestTensor "input1" {0.0 1.0} float32 cpu]
    set input2 [createTestTensor "input2" {0.05 1.05} float32 cpu]
    set result [torch::isclose $input1 $input2 0.1 0.1]
    
    ;# Both should be close with rtol=0.1, atol=0.1
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# NAMED PARAMETER SYNTAX TESTS
#===============================================================================

test isclose-named-2.1 {Basic isclose with named syntax using -input/-other} {
    set input1 [createTestTensor "input1" {1.0 2.0 3.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.0 2.0 3.0} float32 cpu]
    set result [torch::isclose -input $input1 -other $input2]
    
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-named-2.2 {Isclose with named syntax using -tensor1/-tensor2} {
    set input1 [createTestTensor "input1" {1.0 2.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.000001 2.000001} float32 cpu]
    set result [torch::isclose -tensor1 $input1 -tensor2 $input2]
    
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-named-2.3 {Isclose with named syntax - custom tolerances} {
    set input1 [createTestTensor "input1" {1.0 0.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.05 0.05} float32 cpu]
    set result [torch::isclose -input $input1 -other $input2 -rtol 0.1 -atol 0.1]
    
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-named-2.4 {Isclose with named syntax - camelCase parameter names} {
    set input1 [createTestTensor "input1" {1.0 0.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.05 0.05} float32 cpu]
    set result [torch::isclose -input $input1 -other $input2 -relativeTolerance 0.1 -absoluteTolerance 0.1]
    
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-named-2.5 {Isclose with named syntax - mixed parameter order} {
    set input1 [createTestTensor "input1" {1.0 2.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.01 2.01} float32 cpu]
    set result [torch::isclose -rtol 0.05 -other $input2 -input $input1 -atol 0.01]
    
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# CAMELCASE ALIAS TESTS
#===============================================================================

test isclose-camel-3.1 {CamelCase alias isClose functionality} {
    set input1 [createTestTensor "input1" {1.0 2.0 3.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.0 2.0 3.0} float32 cpu]
    set result [torch::isClose -input $input1 -other $input2]
    
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-camel-3.2 {CamelCase alias positional syntax} {
    set input1 [createTestTensor "input1" {1.0 2.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.000001 2.000001} float32 cpu]
    set result [torch::isClose $input1 $input2]
    
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-camel-3.3 {CamelCase alias with all parameters} {
    set input1 [createTestTensor "input1" {1.0 0.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.1 0.1} float32 cpu]
    set result [torch::isClose -input $input1 -other $input2 -rtol 0.15 -atol 0.15 -equal_nan 0]
    
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# DATA TYPE COMPATIBILITY TESTS
#===============================================================================

test isclose-dtype-4.1 {Isclose with different numeric types} {
    set input1 [createTestTensor "input1" {1 2 3} int32 cpu]
    set input2 [createTestTensor "input2" {1 2 3} int32 cpu]
    set result [torch::isclose -input $input1 -other $input2]
    
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-dtype-4.2 {Isclose with float64 tensors} {
    set input1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu -requiresGrad false]
    set input2 [torch::tensor_create -data {1.0000001 2.0000001 3.0000001} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::isclose -input $input1 -other $input2]
    
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-dtype-4.3 {Isclose with same int types} {
    set input1 [createTestTensor "input1" {1 2 3} int32 cpu]
    set input2 [createTestTensor "input2" {1 2 3} int32 cpu]
    set result [torch::isclose -input $input1 -other $input2]
    
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# TOLERANCE PARAMETER TESTS
#===============================================================================

test isclose-tolerance-5.1 {Strict tolerance test} {
    set input1 [createTestTensor "input1" {1.0 2.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.001 2.001} float32 cpu]
    set result [torch::isclose -input $input1 -other $input2 -rtol 1e-6 -atol 1e-6]
    
    ;# Should not be close with very strict tolerance
    set expected [createBoolTensor "expected" {false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-tolerance-5.2 {Loose tolerance test} {
    set input1 [createTestTensor "input1" {1.0 2.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.1 2.1} float32 cpu]
    set result [torch::isclose -input $input1 -other $input2 -rtol 0.2 -atol 0.2]
    
    ;# Should be close with loose tolerance
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-tolerance-5.3 {Absolute tolerance dominates} {
    set input1 [createTestTensor "input1" {0.0 0.0} float32 cpu]
    set input2 [createTestTensor "input2" {0.05 0.08} float32 cpu]
    set result [torch::isclose -input $input1 -other $input2 -rtol 1e-6 -atol 0.1]
    
    ;# Absolute tolerance should dominate for values near zero
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# SPECIAL VALUES TESTS (NaN, Infinity)
#===============================================================================

test isclose-special-6.1 {Isclose with NaN values - equal_nan=false} {
    set input1 [createTestTensor "input1" {1.0 2.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.0 2.0} float32 cpu]
    
    ;# First create NaN by dividing 0/0
    set zero [torch::tensor_create -data {0.0} -dtype float32 -device cpu -requiresGrad false]
    set nan_tensor [torch::tensor_div $zero $zero]
    
    ;# Create tensors with NaN
    set nan1 [torch::tensor_create -data {1.0} -dtype float32 -device cpu -requiresGrad false]
    set nan2 [torch::tensor_create -data {1.0} -dtype float32 -device cpu -requiresGrad false]
    
    set result [torch::isclose -input $input1 -other $input2 -equal_nan 0]
    
    ;# Regular values should be close
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-special-6.2 {Isclose with infinity values} {
    set input1 [createTestTensor "input1" {1.0 2.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.0 2.0} float32 cpu]
    set result [torch::isclose -input $input1 -other $input2]
    
    ;# Identical finite values should be close
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# EDGE CASES AND ERROR HANDLING
#===============================================================================

test isclose-edge-7.1 {Isclose with zero values} {
    set input1 [createTestTensor "input1" {0.0 0.0 0.0} float32 cpu]
    set input2 [createTestTensor "input2" {0.0 0.0 0.0} float32 cpu]
    set result [torch::isclose -input $input1 -other $input2]
    
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-edge-7.2 {Isclose with negative values} {
    set input1 [createTestTensor "input1" {-1.0 -2.0 -3.0} float32 cpu]
    set input2 [createTestTensor "input2" {-1.0 -2.0 -3.0} float32 cpu]
    set result [torch::isclose -input $input1 -other $input2]
    
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-edge-7.3 {Isclose with very small values} {
    set input1 [createTestTensor "input1" {1e-10 2e-10} float32 cpu]
    set input2 [createTestTensor "input2" {1e-10 2e-10} float32 cpu]
    set result [torch::isclose -input $input1 -other $input2]
    
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test isclose-edge-7.4 {Isclose with very large values} {
    set input1 [createTestTensor "input1" {1e10 2e10} float32 cpu]
    set input2 [createTestTensor "input2" {1e10 2e10} float32 cpu]
    set result [torch::isclose -input $input1 -other $input2]
    
    set expected [createBoolTensor "expected" {true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# ERROR HANDLING TESTS
#===============================================================================

test isclose-error-8.1 {Error handling - missing required arguments} {
    set result [catch {torch::isclose} msg]
    set result
} {1}

test isclose-error-8.2 {Error handling - invalid tensor handle} {
    set input1 [createTestTensor "input1" {1.0 2.0} float32 cpu]
    set result [catch {torch::isclose $input1 invalid_tensor} msg]
    set result
} {1}

test isclose-error-8.3 {Error handling - negative tolerance} {
    set input1 [createTestTensor "input1" {1.0 2.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.0 2.0} float32 cpu]
    set result [catch {torch::isclose -input $input1 -other $input2 -rtol -0.1} msg]
    set result
} {1}

test isclose-error-8.4 {Error handling - unknown parameter} {
    set input1 [createTestTensor "input1" {1.0 2.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.0 2.0} float32 cpu]
    set result [catch {torch::isclose -input $input1 -other $input2 -invalid_param 0.1} msg]
    set result
} {1}

#===============================================================================
# CONSISTENCY BETWEEN SYNTAXES
#===============================================================================

test isclose-consistency-9.1 {Consistency between positional and named syntax} {
    set input1 [createTestTensor "input1" {1.0 2.0 3.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.01 2.01 3.01} float32 cpu]
    
    set result_pos [torch::isclose $input1 $input2 0.05 0.05]
    set result_named [torch::isclose -input $input1 -other $input2 -rtol 0.05 -atol 0.05]
    
    set isEqual [tensorsApproxEqual $result_pos $result_named 1e-15]
    set isEqual
} {1}

test isclose-consistency-9.2 {Consistency between snake_case and camelCase} {
    set input1 [createTestTensor "input1" {1.0 2.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.01 2.01} float32 cpu]
    
    set result_snake [torch::isclose -input $input1 -other $input2]
    set result_camel [torch::isClose -input $input1 -other $input2]
    
    set isEqual [tensorsApproxEqual $result_snake $result_camel 1e-15]
    set isEqual
} {1}

cleanupTests 