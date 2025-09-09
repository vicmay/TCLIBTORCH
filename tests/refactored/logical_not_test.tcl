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

test logical_not-positional-1.1 {Basic logical_not with positional syntax - bool tensors} {
    set input [createBoolTensor "input" {true false true false} cpu]
    set result [torch::logical_not $input]
    
    # NOT truth table: NOT true = false, NOT false = true
    set expected [createBoolTensor "expected" {false true false true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_not-positional-1.2 {Logical_not with positional syntax - numeric tensors} {
    set input [createTestTensor "input" {1.0 0.0 2.0 -1.0} float32 cpu]
    set result [torch::logical_not $input]
    
    # Non-zero values are true, zero values are false
    # NOT 1.0 = false, NOT 0.0 = true, NOT 2.0 = false, NOT -1.0 = false
    set expected [createBoolTensor "expected" {false true false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_not-positional-1.3 {Logical_not with positional syntax - all true} {
    set input [createBoolTensor "input" {true true true} cpu]
    set result [torch::logical_not $input]
    
    set expected [createBoolTensor "expected" {false false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_not-positional-1.4 {Logical_not with positional syntax - all false} {
    set input [createBoolTensor "input" {false false false} cpu]
    set result [torch::logical_not $input]
    
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# NAMED PARAMETER SYNTAX TESTS
#===============================================================================

test logical_not-named-2.1 {Basic logical_not with named syntax using -input} {
    set input [createBoolTensor "input" {true false true false} cpu]
    set result [torch::logical_not -input $input]
    
    set expected [createBoolTensor "expected" {false true false true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_not-named-2.2 {Logical_not with named syntax using -tensor} {
    set input [createBoolTensor "input" {false false true true} cpu]
    set result [torch::logical_not -tensor $input]
    
    set expected [createBoolTensor "expected" {true true false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# CAMELCASE ALIAS TESTS
#===============================================================================

test logical_not-camel-3.1 {CamelCase alias logicalNot functionality} {
    set input [createBoolTensor "input" {true false true} cpu]
    set result [torch::logicalNot -input $input]
    
    set expected [createBoolTensor "expected" {false true false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_not-camel-3.2 {CamelCase alias positional syntax} {
    set input [createBoolTensor "input" {false true false} cpu]
    set result [torch::logicalNot $input]
    
    set expected [createBoolTensor "expected" {true false true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# DATA TYPE COMPATIBILITY TESTS
#===============================================================================

test logical_not-dtype-4.1 {Logical_not with different numeric types} {
    set input [createTestTensor "input" {1 0 2 -1} int32 cpu]
    set result [torch::logical_not -input $input]
    
    # 1 = true → false, 0 = false → true, 2 = true → false, -1 = true → false
    set expected [createBoolTensor "expected" {false true false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_not-dtype-4.2 {Logical_not with float64 tensors} {
    set input [torch::tensor_create -data {1.5 0.0 -2.3 0.001} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::logical_not -input $input]
    
    # 1.5 = true → false, 0.0 = false → true, -2.3 = true → false, 0.001 = true → false
    set expected [createBoolTensor "expected" {false true false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_not-dtype-4.3 {Logical_not with integer zeros and ones} {
    set input [createTestTensor "input" {0 1 0 1} int64 cpu]
    set result [torch::logical_not -tensor $input]
    
    # 0 = false → true, 1 = true → false
    set expected [createBoolTensor "expected" {true false true false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# SHAPE AND DIMENSION TESTS
#===============================================================================

test logical_not-shape-5.1 {Logical_not with scalar tensor} {
    set input [createBoolTensor "input" {true} cpu]
    set result [torch::logical_not -input $input]
    
    set expected [createBoolTensor "expected" {false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_not-shape-5.2 {Logical_not with 2D tensor} {
    # Create 2D boolean tensors by creating 1D tensor and reshaping
    set input_1d [torch::tensor_create -data {1 0 0 1} -dtype bool -device cpu -requiresGrad false]
    set input [torch::tensor_reshape $input_1d "2 2"]
    set result [torch::logical_not -input $input]
    
    set expected_1d [torch::tensor_create -data {0 1 1 0} -dtype bool -device cpu -requiresGrad false]
    set expected [torch::tensor_reshape $expected_1d "2 2"]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_not-shape-5.3 {Logical_not with large 1D tensor} {
    set values {}
    set expected_values {}
    for {set i 0} {$i < 100} {incr i} {
        set val [expr {$i % 2 == 0}]
        lappend values $val
        lappend expected_values [expr {!$val}]
    }
    
    set input [createTestTensor "input" $values bool cpu]
    set result [torch::logical_not -tensor $input]
    
    set expected [createTestTensor "expected" $expected_values bool cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# ERROR HANDLING TESTS
#===============================================================================

test logical_not-error-6.1 {Error on missing arguments} {
    set result [catch {torch::logical_not} msg]
    set result
} {1}

test logical_not-error-6.2 {Error on invalid tensor handle} {
    set result [catch {torch::logical_not invalid_handle} msg]
    set result
} {1}

test logical_not-error-6.3 {Error on unknown named parameter} {
    set input [createBoolTensor "input" {true false} cpu]
    set result [catch {torch::logical_not -invalid $input} msg]
    set result
} {1}

test logical_not-error-6.4 {Error on named parameter without value} {
    set result [catch {torch::logical_not -input} msg]
    set result
} {1}

test logical_not-error-6.5 {Error on too many positional arguments} {
    set input1 [createBoolTensor "input1" {true false} cpu]
    set input2 [createBoolTensor "input2" {false true} cpu]
    set result [catch {torch::logical_not $input1 $input2} msg]
    set result
} {1}

test logical_not-error-6.6 {Error on mixed positional and named arguments} {
    set input [createBoolTensor "input" {true false} cpu]
    set result [catch {torch::logical_not $input -input $input} msg]
    set result
} {1}

#===============================================================================
# TRUTH TABLE TESTS
#===============================================================================

test logical_not-truth-7.1 {Complete NOT truth table} {
    set input_true [createBoolTensor "input_true" {true} cpu]
    set input_false [createBoolTensor "input_false" {false} cpu]
    
    set result_true [torch::logical_not -input $input_true]
    set result_false [torch::logical_not -input $input_false]
    
    set expected_true [createBoolTensor "expected_true" {false} cpu]
    set expected_false [createBoolTensor "expected_false" {true} cpu]
    
    set equal1 [tensorsApproxEqual $result_true $expected_true 1e-15]
    set equal2 [tensorsApproxEqual $result_false $expected_false 1e-15]
    
    expr {$equal1 && $equal2}
} {1}

test logical_not-truth-7.2 {Numeric truth table equivalents} {
    set input_nonzero [createTestTensor "input_nonzero" {1.0 -1.0 2.5 0.1} float32 cpu]
    set input_zero [createTestTensor "input_zero" {0.0 0.0 0.0 0.0} float32 cpu]
    
    set result_nonzero [torch::logical_not -tensor $input_nonzero]
    set result_zero [torch::logical_not -tensor $input_zero]
    
    # Non-zero → false, Zero → true
    set expected_nonzero [createBoolTensor "expected_nonzero" {false false false false} cpu]
    set expected_zero [createBoolTensor "expected_zero" {true true true true} cpu]
    
    set equal1 [tensorsApproxEqual $result_nonzero $expected_nonzero 1e-15]
    set equal2 [tensorsApproxEqual $result_zero $expected_zero 1e-15]
    
    expr {$equal1 && $equal2}
} {1}

#===============================================================================
# CONSISTENCY TESTS
#===============================================================================

test logical_not-consistency-8.1 {Consistency between positional and named syntax} {
    set input [createBoolTensor "input" {true false true false} cpu]
    
    set result_positional [torch::logical_not $input]
    set result_named [torch::logical_not -input $input]
    
    set isEqual [tensorsApproxEqual $result_positional $result_named 1e-15]
    set isEqual
} {1}

test logical_not-consistency-8.2 {Consistency with different parameter names} {
    set input [createBoolTensor "input" {false true false} cpu]
    
    set result_input [torch::logical_not -input $input]
    set result_tensor [torch::logical_not -tensor $input]
    
    set isEqual [tensorsApproxEqual $result_input $result_tensor 1e-15]
    set isEqual
} {1}

test logical_not-consistency-8.3 {Consistency between snake_case and camelCase} {
    set input [createBoolTensor "input" {true true false false} cpu]
    
    set result_snake [torch::logical_not -input $input]
    set result_camel [torch::logicalNot -input $input]
    
    set isEqual [tensorsApproxEqual $result_snake $result_camel 1e-15]
    set isEqual
} {1}

#===============================================================================
# COMPLEX OPERATIONS TESTS
#===============================================================================

test logical_not-complex-9.1 {Double negation should return original} {
    set input [createBoolTensor "input" {true false true false} cpu]
    
    set result1 [torch::logical_not $input]
    set result2 [torch::logical_not $result1]
    
    set isEqual [tensorsApproxEqual $result2 $input 1e-15]
    set isEqual
} {1}

test logical_not-complex-9.2 {Logical_not combined with logical_and} {
    set a [createBoolTensor "a" {true false} cpu]
    set b [createBoolTensor "b" {true true} cpu]
    
    # De Morgan's law: NOT(a AND b) = (NOT a) OR (NOT b)
    set and_result [torch::logical_and $a $b]
    set not_and [torch::logical_not $and_result]
    
    set not_a [torch::logical_not $a]
    set not_b [torch::logical_not $b]
    set or_result [torch::logical_or $not_a $not_b]
    
    set isEqual [tensorsApproxEqual $not_and $or_result 1e-15]
    set isEqual
} {1}

test logical_not-complex-9.3 {Chained operations compatibility} {
    set input [createBoolTensor "input" {true false true false} cpu]
    
    # Test: NOT(NOT(NOT(input))) = NOT(input)
    set result1 [torch::logical_not $input]
    set result2 [torch::logical_not $result1]
    set result3 [torch::logical_not $result2]
    
    set expected [torch::logical_not $input]
    set isEqual [tensorsApproxEqual $result3 $expected 1e-15]
    
    set isEqual
} {1}

cleanupTests 