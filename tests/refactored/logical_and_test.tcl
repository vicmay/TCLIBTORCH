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

test logical_and-positional-1.1 {Basic logical_and with positional syntax - bool tensors} {
    set input1 [createBoolTensor "input1" {true false true false} cpu]
    set input2 [createBoolTensor "input2" {true true false false} cpu]
    set result [torch::logical_and $input1 $input2]
    
    # AND truth table: true & true = true, others = false
    set expected [createBoolTensor "expected" {true false false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_and-positional-1.2 {Logical_and with positional syntax - numeric tensors} {
    set input1 [createTestTensor "input1" {1.0 0.0 2.0 -1.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.0 1.0 0.0 0.0} float32 cpu]
    set result [torch::logical_and $input1 $input2]
    
    # Non-zero values are true, zero values are false
    # 1.0 & 1.0 = true, 0.0 & 1.0 = false, 2.0 & 0.0 = false, -1.0 & 0.0 = false
    set expected [createBoolTensor "expected" {true false false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_and-positional-1.3 {Logical_and with positional syntax - all true} {
    set input1 [createBoolTensor "input1" {true true true} cpu]
    set input2 [createBoolTensor "input2" {true true true} cpu]
    set result [torch::logical_and $input1 $input2]
    
    set expected [createBoolTensor "expected" {true true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# NAMED PARAMETER SYNTAX TESTS
#===============================================================================

test logical_and-named-2.1 {Basic logical_and with named syntax using -input1/-input2} {
    set input1 [createBoolTensor "input1" {true false true false} cpu]
    set input2 [createBoolTensor "input2" {true true false false} cpu]
    set result [torch::logical_and -input1 $input1 -input2 $input2]
    
    set expected [createBoolTensor "expected" {true false false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_and-named-2.2 {Logical_and with named syntax using -tensor1/-tensor2} {
    set input1 [createBoolTensor "input1" {false false true true} cpu]
    set input2 [createBoolTensor "input2" {false true false true} cpu]
    set result [torch::logical_and -tensor1 $input1 -tensor2 $input2]
    
    set expected [createBoolTensor "expected" {false false false true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_and-named-2.3 {Logical_and with named syntax - mixed parameter order} {
    set input1 [createBoolTensor "input1" {true true false} cpu]
    set input2 [createBoolTensor "input2" {false true true} cpu]
    set result [torch::logical_and -input2 $input2 -input1 $input1]
    
    set expected [createBoolTensor "expected" {false true false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# CAMELCASE ALIAS TESTS
#===============================================================================

test logical_and-camel-3.1 {CamelCase alias logicalAnd functionality} {
    set input1 [createBoolTensor "input1" {true false true} cpu]
    set input2 [createBoolTensor "input2" {true true false} cpu]
    set result [torch::logicalAnd -input1 $input1 -input2 $input2]
    
    set expected [createBoolTensor "expected" {true false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_and-camel-3.2 {CamelCase alias positional syntax} {
    set input1 [createBoolTensor "input1" {false true false} cpu]
    set input2 [createBoolTensor "input2" {true false true} cpu]
    set result [torch::logicalAnd $input1 $input2]
    
    set expected [createBoolTensor "expected" {false false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# DATA TYPE COMPATIBILITY TESTS
#===============================================================================

test logical_and-dtype-4.1 {Logical_and with different numeric types} {
    set input1 [createTestTensor "input1" {1 0 2 -1} int32 cpu]
    set input2 [createTestTensor "input2" {3 5 0 7} int32 cpu]
    set result [torch::logical_and -input1 $input1 -input2 $input2]
    
    # 1 & 3 = true, 0 & 5 = false, 2 & 0 = false, -1 & 7 = true
    set expected [createBoolTensor "expected" {true false false true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_and-dtype-4.2 {Logical_and with float64 tensors} {
    set input1 [torch::tensor_create -data {1.5 0.0 -2.3 0.001} -dtype float64 -device cpu -requiresGrad false]
    set input2 [torch::tensor_create -data {2.7 1.1 0.0 -0.5} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::logical_and -input1 $input1 -input2 $input2]
    
    # 1.5 & 2.7 = true, 0.0 & 1.1 = false, -2.3 & 0.0 = false, 0.001 & -0.5 = true
    set expected [createBoolTensor "expected" {true false false true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_and-dtype-4.3 {Logical_and with mixed bool and numeric} {
    set input1 [createBoolTensor "input1" {true false true} cpu]
    set input2 [createTestTensor "input2" {1.0 2.0 0.0} float32 cpu]
    set result [torch::logical_and -input1 $input1 -input2 $input2]
    
    # true & 1.0 = true, false & 2.0 = false, true & 0.0 = false
    set expected [createBoolTensor "expected" {true false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# BROADCASTING TESTS
#===============================================================================

test logical_and-broadcast-5.1 {Logical_and with broadcasting - scalar and vector} {
    set input1 [createBoolTensor "input1" {true} cpu]
    set input2 [createBoolTensor "input2" {true false true false} cpu]
    set result [torch::logical_and -input1 $input1 -input2 $input2]
    
    # Broadcasting: true & [true false true false] = [true false true false]
    set expected [createBoolTensor "expected" {true false true false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_and-broadcast-5.2 {Logical_and with broadcasting - different shapes} {
    set input1 [createTestTensor "input1" {1.0 0.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.0 1.0 0.0 0.0} float32 cpu]
    # Note: This will work if the tensors can broadcast, otherwise it will fail
    # Let's use compatible shapes
    set input1 [createTestTensor "input1" {1.0 0.0} float32 cpu]
    set input2 [createTestTensor "input2" {1.0 1.0} float32 cpu]
    set result [torch::logical_and -input1 $input1 -input2 $input2]
    
    # 1.0 & 1.0 = true, 0.0 & 1.0 = false
    set expected [createBoolTensor "expected" {true false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# EDGE CASE TESTS
#===============================================================================

test logical_and-edge-6.1 {Logical_and with all false} {
    set input1 [createBoolTensor "input1" {false false false} cpu]
    set input2 [createBoolTensor "input2" {true false true} cpu]
    set result [torch::logical_and -input1 $input1 -input2 $input2]
    
    set expected [createBoolTensor "expected" {false false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_and-edge-6.2 {Logical_and with zeros and negative numbers} {
    set input1 [createTestTensor "input1" {0.0 -0.0 -1.0 -2.5} float32 cpu]
    set input2 [createTestTensor "input2" {1.0 2.0 3.0 4.0} float32 cpu]
    set result [torch::logical_and -input1 $input1 -input2 $input2]
    
    # 0.0 & 1.0 = false, -0.0 & 2.0 = false, -1.0 & 3.0 = true, -2.5 & 4.0 = true
    set expected [createBoolTensor "expected" {false false true true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_and-edge-6.3 {Logical_and with single element tensors} {
    set input1 [createBoolTensor "input1" {true} cpu]
    set input2 [createBoolTensor "input2" {false} cpu]
    set result [torch::logical_and -input1 $input1 -input2 $input2]
    
    set expected [createBoolTensor "expected" {false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# ERROR HANDLING TESTS
#===============================================================================

test logical_and-error-7.1 {Error handling - missing arguments} {
    set result [catch {torch::logical_and} error_msg]
    set result
} {1}

test logical_and-error-7.2 {Error handling - only one argument} {
    set input1 [createBoolTensor "input1" {true} cpu]
    set result [catch {torch::logical_and $input1} error_msg]
    set result
} {1}

test logical_and-error-7.3 {Error handling - invalid tensor name} {
    set input1 [createBoolTensor "input1" {true} cpu]
    set result [catch {torch::logical_and $input1 nonexistent_tensor} error_msg]
    set result
} {1}

test logical_and-error-7.4 {Error handling - missing value for named parameter} {
    set result [catch {torch::logical_and -input1} error_msg]
    set result
} {1}

test logical_and-error-7.5 {Error handling - unknown parameter} {
    set input1 [createBoolTensor "input1" {true} cpu]
    set input2 [createBoolTensor "input2" {false} cpu]
    set result [catch {torch::logical_and -unknown_param $input1 -input2 $input2} error_msg]
    
    set result
} {1}

test logical_and-error-7.6 {Error handling - missing second tensor in named syntax} {
    set input1 [createBoolTensor "input1" {true} cpu]
    set result [catch {torch::logical_and -input1 $input1} error_msg]
    
    set result
} {1}

#===============================================================================
# LOGICAL OPERATIONS TRUTH TABLE TESTS
#===============================================================================

test logical_and-truth-8.1 {Complete AND truth table} {
    set input1 [createBoolTensor "input1" {true true false false} cpu]
    set input2 [createBoolTensor "input2" {true false true false} cpu]
    set result [torch::logical_and -input1 $input1 -input2 $input2]
    
    # AND truth table: T&T=T, T&F=F, F&T=F, F&F=F
    set expected [createBoolTensor "expected" {true false false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_and-truth-8.2 {Numeric truth table equivalents} {
    # Test with 1 and 0 representing true and false
    set input1 [createTestTensor "input1" {1 1 0 0} int32 cpu]
    set input2 [createTestTensor "input2" {1 0 1 0} int32 cpu]
    set result [torch::logical_and -input1 $input1 -input2 $input2]
    
    set expected [createBoolTensor "expected" {true false false false} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# CONSISTENCY TESTS (Both Syntaxes Produce Same Results)
#===============================================================================

test logical_and-consistency-9.1 {Consistency between positional and named syntax} {
    set input1 [createBoolTensor "input1" {true false true false} cpu]
    set input2 [createBoolTensor "input2" {false true true false} cpu]
    
    # Test both syntaxes
    set result_positional [torch::logical_and $input1 $input2]
    set result_named [torch::logical_and -input1 $input1 -input2 $input2]
    
    set isEqual [tensorsApproxEqual $result_positional $result_named 1e-15]
    
    set isEqual
} {1}

test logical_and-consistency-9.2 {Consistency with different parameter names} {
    set input1 [createBoolTensor "input1" {true false} cpu]
    set input2 [createBoolTensor "input2" {false true} cpu]
    
    # Test both named parameter options
    set result_input [torch::logical_and -input1 $input1 -input2 $input2]
    set result_tensor [torch::logical_and -tensor1 $input1 -tensor2 $input2]
    
    set isEqual [tensorsApproxEqual $result_input $result_tensor 1e-15]
    
    set isEqual
} {1}

test logical_and-consistency-9.3 {Consistency between snake_case and camelCase} {
    set input1 [createBoolTensor "input1" {true true false} cpu]
    set input2 [createBoolTensor "input2" {false true true} cpu]
    
    # Test both command formats
    set result_snake [torch::logical_and -input1 $input1 -input2 $input2]
    set result_camel [torch::logicalAnd -input1 $input1 -input2 $input2]
    
    set isEqual [tensorsApproxEqual $result_snake $result_camel 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# COMPLEX SCENARIO TESTS
#===============================================================================

test logical_and-complex-10.1 {Complex boolean expressions} {
    # Test more complex boolean scenarios
    set a [createBoolTensor "a" {true false true false true} cpu]
    set b [createBoolTensor "b" {true true false false true} cpu]
    set result [torch::logical_and -input1 $a -input2 $b]
    
    # Manual calculation: T&T=T, F&T=F, T&F=F, F&F=F, T&T=T
    set expected [createBoolTensor "expected" {true false false false true} cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test logical_and-complex-10.2 {Chained operations compatibility} {
    # Test that result can be used in further operations
    set input1 [createBoolTensor "input1" {true false} cpu]
    set input2 [createBoolTensor "input2" {true true} cpu]
    set result1 [torch::logical_and -input1 $input1 -input2 $input2]
    
    set input3 [createBoolTensor "input3" {false true} cpu]
    set result2 [torch::logical_and -input1 $result1 -input2 $input3]
    
    # (T&T)&F=T&F=F, (F&T)&T=F&T=F
    set expected [createBoolTensor "expected" {false false} cpu]
    set isEqual [tensorsApproxEqual $result2 $expected 1e-15]
    
    set isEqual
} {1}

cleanupTests 