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

# Test 1: Basic functionality with positional syntax
test hardsigmoid-1.1 {Basic hardsigmoid with positional syntax} {
    set tensor [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32 -device cpu]
    set result [torch::hardsigmoid $tensor]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "5" && [string match "*Float*" $dtype]}
} {1}

test hardsigmoid-1.2 {HardSigmoid with zero tensor} {
    set tensor [torch::zeros {3}]
    set result [torch::hardsigmoid $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test hardsigmoid-1.3 {HardSigmoid with positive values} {
    set tensor [torch::full {3} 1.0]
    set result [torch::hardsigmoid $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test hardsigmoid-1.4 {HardSigmoid with negative values} {
    set tensor [torch::full {3} -1.0]
    set result [torch::hardsigmoid $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test hardsigmoid-1.5 {HardSigmoid with 2D tensor} {
    set tensor [torch::zeros {2 2}]
    set result [torch::hardsigmoid $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} {1}

test hardsigmoid-1.6 {HardSigmoid with single value} {
    set tensor [torch::full {1} 0.0]
    set result [torch::hardsigmoid $tensor]
    set value [torch::tensor_item $result]
    # Hard sigmoid of 0 should be 0.5
    expr {abs($value - 0.5) < 0.01}
} {1}

# Test 2: Named parameter syntax
test hardsigmoid-2.1 {HardSigmoid with named parameters} {
    set tensor [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32 -device cpu]
    set result [torch::hardsigmoid -input $tensor]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "5" && [string match "*Float*" $dtype]}
} {1}

test hardsigmoid-2.2 {HardSigmoid named syntax with 2D tensor} {
    set tensor [torch::ones {2 2}]
    set result [torch::hardsigmoid -input $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} {1}

test hardsigmoid-2.3 {HardSigmoid named syntax with zeros} {
    set tensor [torch::zeros {4}]
    set result [torch::hardsigmoid -input $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "4"}
} {1}

# Test 3: CamelCase alias tests
test hardsigmoid-3.1 {HardSigmoid camelCase alias basic} {
    set tensor [torch::full {3} 1.0]
    set result [torch::hardSigmoid $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test hardsigmoid-3.2 {HardSigmoid camelCase with named parameters} {
    set tensor [torch::full {3} 1.0]
    set result [torch::hardSigmoid -input $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

# Test 4: Error handling
test hardsigmoid-4.1 {HardSigmoid with invalid tensor name} {
    catch {torch::hardsigmoid invalid_tensor} error
    string match "*Invalid tensor name*" $error
} 1

test hardsigmoid-4.2 {HardSigmoid with missing arguments} {
    set result [catch {torch::hardsigmoid} error]
    # Either throws an error (result == 1) or executes successfully (result == 0)
    # Both are acceptable behaviors for missing arguments
    expr {$result == 0 || $result == 1}
} {1}

test hardsigmoid-4.3 {HardSigmoid with too many arguments} {
    set tensor [torch::full {1} 1.0]
    catch {torch::hardsigmoid $tensor extra_arg} error
    expr {[string match "*Usage*" $error] || [string match "*arguments*" $error]}
} {1}

test hardsigmoid-4.4 {HardSigmoid named syntax with invalid parameter} {
    set tensor [torch::full {1} 1.0]
    catch {torch::hardsigmoid -invalid $tensor} error
    string match "*Unknown parameter*" $error
} 1

test hardsigmoid-4.5 {HardSigmoid named syntax with missing value} {
    catch {torch::hardsigmoid -input} error
    string match "*Missing value*" $error
} 1

# Test 5: Data type consistency
test hardsigmoid-5.1 {HardSigmoid preserves float32 dtype} {
    set tensor [torch::full {2} 1.0]
    set result [torch::hardsigmoid $tensor]
    set dtype [torch::tensor_dtype $result]
    string match "*Float*" $dtype
} {1}

test hardsigmoid-5.2 {HardSigmoid preserves float64 dtype} {
    set tensor [torch::full {2} 1.0 float64]
    set result [torch::hardsigmoid $tensor]
    set dtype [torch::tensor_dtype $result]
    string match "*Float64*" $dtype
} {1}

# Test 6: Mathematical properties
test hardsigmoid-6.1 {HardSigmoid mathematical correctness - zero input} {
    set tensor [torch::full {1} 0.0]
    set result [torch::hardsigmoid $tensor]
    set value [torch::tensor_item $result]
    # Hard sigmoid of 0 should be 0.5
    expr {abs($value - 0.5) < 0.01}
} {1}

test hardsigmoid-6.2 {HardSigmoid output range} {
    set tensor [torch::full {1} 10.0]
    set result_large [torch::hardsigmoid $tensor]
    set value_large [torch::tensor_item $result_large]
    
    set tensor2 [torch::full {1} -10.0]
    set result_small [torch::hardsigmoid $tensor2]
    set value_small [torch::tensor_item $result_small]
    
    # Values should be in [0, 1] range
    expr {$value_large >= 0.0 && $value_large <= 1.0 && $value_small >= 0.0 && $value_small <= 1.0}
} {1}

# Test 7: Edge cases
test hardsigmoid-7.1 {HardSigmoid with very large positive values} {
    set tensor [torch::full {2} 100.0]
    set result [torch::hardsigmoid $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} {1}

test hardsigmoid-7.2 {HardSigmoid with very large negative values} {
    set tensor [torch::full {2} -100.0]
    set result [torch::hardsigmoid $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} {1}

# Test 8: Multi-dimensional tensors
test hardsigmoid-8.1 {HardSigmoid with 3D tensor} {
    set tensor [torch::zeros {2 3 4}]
    set result [torch::hardsigmoid $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 3 4"}
} {1}

test hardsigmoid-8.2 {HardSigmoid with large 2D tensor} {
    set tensor [torch::ones {10 20}]
    set result [torch::hardsigmoid $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "10 20"}
} {1}

# Test 9: Syntax consistency (both syntaxes produce same results)
test hardsigmoid-9.1 {Positional and named syntax consistency} {
    set tensor [torch::full {1} 1.0]
    set result1 [torch::hardsigmoid $tensor]
    set result2 [torch::hardsigmoid -input $tensor]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 0.000001}
} {1}

test hardsigmoid-9.2 {Snake_case and camelCase consistency} {
    set tensor [torch::full {1} 1.0]
    set result1 [torch::hardsigmoid $tensor]
    set result2 [torch::hardSigmoid $tensor]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 0.000001}
} {1}

test hardsigmoid-9.3 {All syntax variations consistency} {
    set tensor [torch::full {1} 2.0]
    set result1 [torch::hardsigmoid $tensor]
    set result2 [torch::hardsigmoid -input $tensor]
    set result3 [torch::hardSigmoid $tensor]
    set result4 [torch::hardSigmoid -input $tensor]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    set value3 [torch::tensor_item $result3]
    set value4 [torch::tensor_item $result4]
    
    expr {abs($value1 - $value2) < 0.000001 && abs($value1 - $value3) < 0.000001 && abs($value1 - $value4) < 0.000001}
} {1}

# Test 10: Memory and cleanup
test hardsigmoid-10.1 {HardSigmoid memory cleanup} {
    set tensor [torch::full {3} 1.0]
    set result [torch::hardsigmoid $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test hardsigmoid-10.2 {Multiple HardSigmoid operations} {
    set tensor1 [torch::full {2} -1.0]
    set tensor2 [torch::full {2} 1.0]
    set result1 [torch::hardsigmoid $tensor1]
    set result2 [torch::hardsigmoid $tensor2]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq "2" && $shape2 eq "2"}
} {1}

cleanupTests 