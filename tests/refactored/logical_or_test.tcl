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

test logical_or-positional-1.1 {Basic logical_or with positional syntax - bool tensors} {
    set input1 [createBoolTensor "input1" {true false true false} cpu]
    set input2 [createBoolTensor "input2" {true true false false} cpu]
    set result [torch::logical_or $input1 $input2]
    
    # OR truth table: true OR anything = true, false OR false = false
    set expected [createBoolTensor "expected" {true true true false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_or-positional-1.2 {Logical_or with positional syntax - numeric tensors} {
    set input1 [createTestTensor "input1" {1.0 0.0 2.0 0.0} float32 cpu]
    set input2 [createTestTensor "input2" {0.0 1.0 0.0 0.0} float32 cpu]
    set result [torch::logical_or $input1 $input2]
    
    # 1.0 OR 0.0 = true, 0.0 OR 1.0 = true, 2.0 OR 0.0 = true, 0.0 OR 0.0 = false
    set expected [createBoolTensor "expected" {true true true false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_or-positional-1.3 {Logical_or with positional syntax - all true} {
    set input1 [createBoolTensor "input1" {true true true} cpu]
    set input2 [createBoolTensor "input2" {true false true} cpu]
    set result [torch::logical_or $input1 $input2]
    
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_or-positional-1.4 {Logical_or with positional syntax - all false} {
    set input1 [createBoolTensor "input1" {false false false} cpu]
    set input2 [createBoolTensor "input2" {false false false} cpu]
    set result [torch::logical_or $input1 $input2]
    
    set expected [createBoolTensor "expected" {false false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# NAMED PARAMETER SYNTAX TESTS
#===============================================================================

test logical_or-named-2.1 {Basic logical_or with named syntax using -input1/-input2} {
    set input1 [createBoolTensor "input1" {true false true false} cpu]
    set input2 [createBoolTensor "input2" {false true false true} cpu]
    set result [torch::logical_or -input1 $input1 -input2 $input2]
    
    set expected [createBoolTensor "expected" {true true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_or-named-2.2 {Logical_or with named syntax using -tensor1/-tensor2} {
    set input1 [createBoolTensor "input1" {false false true true} cpu]
    set input2 [createBoolTensor "input2" {false true false true} cpu]
    set result [torch::logical_or -tensor1 $input1 -tensor2 $input2]
    
    set expected [createBoolTensor "expected" {false true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_or-named-2.3 {Logical_or with mixed parameter names} {
    set input1 [createBoolTensor "input1" {true false} cpu]
    set input2 [createBoolTensor "input2" {false false} cpu]
    set result [torch::logical_or -input1 $input1 -tensor2 $input2]
    
    set expected [createBoolTensor "expected" {true false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# CAMELCASE ALIAS TESTS
#===============================================================================

test logical_or-camel-3.1 {CamelCase alias logicalOr functionality} {
    set input1 [createBoolTensor "input1" {true false true} cpu]
    set input2 [createBoolTensor "input2" {false true false} cpu]
    set result [torch::logicalOr -input1 $input1 -input2 $input2]
    
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_or-camel-3.2 {CamelCase alias positional syntax} {
    set input1 [createBoolTensor "input1" {false true false} cpu]
    set input2 [createBoolTensor "input2" {true false false} cpu]
    set result [torch::logicalOr $input1 $input2]
    
    set expected [createBoolTensor "expected" {true true false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# DATA TYPE COMPATIBILITY TESTS
#===============================================================================

test logical_or-dtype-4.1 {Logical_or with different numeric types} {
    set input1 [createTestTensor "input1" {1 0 2 -1} int32 cpu]
    set input2 [createTestTensor "input2" {0 1 0 0} int32 cpu]
    set result [torch::logical_or -input1 $input1 -input2 $input2]
    
    # 1 OR 0 = true, 0 OR 1 = true, 2 OR 0 = true, -1 OR 0 = true
    set expected [createBoolTensor "expected" {true true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_or-dtype-4.2 {Logical_or with float64 tensors} {
    set input1 [torch::tensor_create -data {1.5 0.0 -2.3 0.0} -dtype float64 -device cpu -requiresGrad false]
    set input2 [torch::tensor_create -data {0.0 0.0 0.0 0.001} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::logical_or -input1 $input1 -input2 $input2]
    
    # 1.5 OR 0.0 = true, 0.0 OR 0.0 = false, -2.3 OR 0.0 = true, 0.0 OR 0.001 = true
    set expected [createBoolTensor "expected" {true false true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_or-dtype-4.3 {Logical_or with integer zeros and ones} {
    set input1 [createTestTensor "input1" {0 1 0 1} int64 cpu]
    set input2 [createTestTensor "input2" {1 0 0 1} int64 cpu]
    set result [torch::logical_or -tensor1 $input1 -tensor2 $input2]
    
    # 0 OR 1 = true, 1 OR 0 = true, 0 OR 0 = false, 1 OR 1 = true
    set expected [createBoolTensor "expected" {true true false true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# BROADCASTING TESTS
#===============================================================================

test logical_or-broadcast-5.1 {Broadcasting with different shapes} {
    set input1 [createBoolTensor "input1" {true false} cpu]
    set input2 [createBoolTensor "input2" {false} cpu]
    set result [torch::logical_or -input1 $input1 -input2 $input2]
    
    # Broadcasting: [true, false] OR [false] = [true, false]
    set expected [createBoolTensor "expected" {true false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_or-broadcast-5.2 {Broadcasting with scalar} {
    set input1 [createBoolTensor "input1" {false false false} cpu]
    set input2 [createBoolTensor "input2" {true} cpu]
    set result [torch::logical_or -tensor1 $input1 -tensor2 $input2]
    
    # Broadcasting: [false, false, false] OR [true] = [true, true, true]
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# ERROR HANDLING TESTS
#===============================================================================

test logical_or-error-6.1 {Error on missing arguments} {
    set result [catch {torch::logical_or} msg]
    set result
} {1}

test logical_or-error-6.2 {Error on single argument} {
    set input [createBoolTensor "input" {true false} cpu]
    set result [catch {torch::logical_or $input} msg]
    set result
} {1}

test logical_or-error-6.3 {Error on invalid first tensor handle} {
    set input2 [createBoolTensor "input2" {true false} cpu]
    set result [catch {torch::logical_or invalid_handle $input2} msg]
    set result
} {1}

test logical_or-error-6.4 {Error on invalid second tensor handle} {
    set input1 [createBoolTensor "input1" {true false} cpu]
    set result [catch {torch::logical_or $input1 invalid_handle} msg]
    set result
} {1}

test logical_or-error-6.5 {Error on unknown named parameter} {
    set input1 [createBoolTensor "input1" {true false} cpu]
    set input2 [createBoolTensor "input2" {false true} cpu]
    set result [catch {torch::logical_or -invalid $input1 -input2 $input2} msg]
    set result
} {1}

test logical_or-error-6.6 {Error on named parameter without value} {
    set input1 [createBoolTensor "input1" {true false} cpu]
    set result [catch {torch::logical_or -input1 $input1 -input2} msg]
    set result
} {1}

test logical_or-error-6.7 {Error on missing required parameter} {
    set input1 [createBoolTensor "input1" {true false} cpu]
    set result [catch {torch::logical_or -input1 $input1} msg]
    set result
} {1}

#===============================================================================
# TRUTH TABLE TESTS
#===============================================================================

test logical_or-truth-7.1 {Complete OR truth table} {
    set input1_ff [createBoolTensor "input1_ff" {false false} cpu]
    set input1_ft [createBoolTensor "input1_ft" {false true} cpu]
    set input1_tf [createBoolTensor "input1_tf" {true false} cpu]
    set input1_tt [createBoolTensor "input1_tt" {true true} cpu]
    
    set input2_ff [createBoolTensor "input2_ff" {false false} cpu]
    set input2_ft [createBoolTensor "input2_ft" {false true} cpu]
    set input2_tf [createBoolTensor "input2_tf" {true false} cpu]
    set input2_tt [createBoolTensor "input2_tt" {true true} cpu]
    
    set result_ff [torch::logical_or -input1 $input1_ff -input2 $input2_ff]
    set result_ft [torch::logical_or -input1 $input1_ft -input2 $input2_ft]
    set result_tf [torch::logical_or -input1 $input1_tf -input2 $input2_tf]
    set result_tt [torch::logical_or -input1 $input1_tt -input2 $input2_tt]
    
    set expected_ff [createBoolTensor "expected_ff" {false false} cpu]
    set expected_ft [createBoolTensor "expected_ft" {false true} cpu]
    set expected_tf [createBoolTensor "expected_tf" {true false} cpu]
    set expected_tt [createBoolTensor "expected_tt" {true true} cpu]
    
    set equal_ff [tensorsApproxEqual $result_ff $expected_ff 1e-15]
    set equal_ft [tensorsApproxEqual $result_ft $expected_ft 1e-15]
    set equal_tf [tensorsApproxEqual $result_tf $expected_tf 1e-15]
    set equal_tt [tensorsApproxEqual $result_tt $expected_tt 1e-15]
    
    expr {$equal_ff && $equal_ft && $equal_tf && $equal_tt}
} {1}

test logical_or-truth-7.2 {Numeric truth table equivalents} {
    set input1_zero [createTestTensor "input1_zero" {0.0 0.0} float32 cpu]
    set input1_nonzero [createTestTensor "input1_nonzero" {1.0 -1.0} float32 cpu]
    
    set input2_zero [createTestTensor "input2_zero" {0.0 0.0} float32 cpu]
    set input2_nonzero [createTestTensor "input2_nonzero" {2.0 -2.0} float32 cpu]
    
    set result_zz [torch::logical_or -tensor1 $input1_zero -tensor2 $input2_zero]
    set result_zn [torch::logical_or -tensor1 $input1_zero -tensor2 $input2_nonzero]
    set result_nz [torch::logical_or -tensor1 $input1_nonzero -tensor2 $input2_zero]
    set result_nn [torch::logical_or -tensor1 $input1_nonzero -tensor2 $input2_nonzero]
    
    # 0 OR 0 = false, 0 OR nonzero = true, nonzero OR 0 = true, nonzero OR nonzero = true
    set expected_zz [createBoolTensor "expected_zz" {false false} cpu]
    set expected_zn [createBoolTensor "expected_zn" {true true} cpu]
    set expected_nz [createBoolTensor "expected_nz" {true true} cpu]
    set expected_nn [createBoolTensor "expected_nn" {true true} cpu]
    
    set equal_zz [tensorsApproxEqual $result_zz $expected_zz 1e-15]
    set equal_zn [tensorsApproxEqual $result_zn $expected_zn 1e-15]
    set equal_nz [tensorsApproxEqual $result_nz $expected_nz 1e-15]
    set equal_nn [tensorsApproxEqual $result_nn $expected_nn 1e-15]
    
    expr {$equal_zz && $equal_zn && $equal_nz && $equal_nn}
} {1}

#===============================================================================
# CONSISTENCY TESTS
#===============================================================================

test logical_or-consistency-8.1 {Consistency between positional and named syntax} {
    set input1 [createBoolTensor "input1" {true false true false} cpu]
    set input2 [createBoolTensor "input2" {false true false true} cpu]
    
    set result_positional [torch::logical_or $input1 $input2]
    set result_named [torch::logical_or -input1 $input1 -input2 $input2]
    
    set isEqual [tensorsApproxEqual $result_positional $result_named 1e-15]
    set isEqual
} {1}

test logical_or-consistency-8.2 {Consistency with different parameter names} {
    set input1 [createBoolTensor "input1" {false true false} cpu]
    set input2 [createBoolTensor "input2" {true false true} cpu]
    
    set result_input [torch::logical_or -input1 $input1 -input2 $input2]
    set result_tensor [torch::logical_or -tensor1 $input1 -tensor2 $input2]
    
    set isEqual [tensorsApproxEqual $result_input $result_tensor 1e-15]
    set isEqual
} {1}

test logical_or-consistency-8.3 {Consistency between snake_case and camelCase} {
    set input1 [createBoolTensor "input1" {true true false false} cpu]
    set input2 [createBoolTensor "input2" {true false true false} cpu]
    
    set result_snake [torch::logical_or -input1 $input1 -input2 $input2]
    set result_camel [torch::logicalOr -input1 $input1 -input2 $input2]
    
    set isEqual [tensorsApproxEqual $result_snake $result_camel 1e-15]
    set isEqual
} {1}

#===============================================================================
# COMPLEX OPERATIONS TESTS
#===============================================================================

test logical_or-complex-9.1 {OR with identity: A OR false = A} {
    set input_a [createBoolTensor "input_a" {true false true false} cpu]
    set input_false [createBoolTensor "input_false" {false false false false} cpu]
    
    set result [torch::logical_or $input_a $input_false]
    
    set isEqual [tensorsApproxEqual $result $input_a 1e-15]
    set isEqual
} {1}

test logical_or-complex-9.2 {OR with annihilator: A OR true = true} {
    set input_a [createBoolTensor "input_a" {true false true false} cpu]
    set input_true [createBoolTensor "input_true" {true true true true} cpu]
    
    set result [torch::logical_or $input_a $input_true]
    
    set isEqual [tensorsApproxEqual $result $input_true 1e-15]
    set isEqual
} {1}

test logical_or-complex-9.3 {Commutative property: A OR B = B OR A} {
    set input_a [createBoolTensor "input_a" {true false true false} cpu]
    set input_b [createBoolTensor "input_b" {false true false true} cpu]
    
    set result_ab [torch::logical_or -input1 $input_a -input2 $input_b]
    set result_ba [torch::logical_or -input1 $input_b -input2 $input_a]
    
    set isEqual [tensorsApproxEqual $result_ab $result_ba 1e-15]
    set isEqual
} {1}

cleanupTests 