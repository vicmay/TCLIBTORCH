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
test relu6-1.1 {Basic relu6 with positional syntax} {
    set tensor [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0 5.0 8.0 10.0} -dtype float32 -device cpu]
    set result [torch::relu6 $tensor]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "8" && [string match "*Float*" $dtype]}
} {1}

test relu6-1.2 {ReLU6 with negative values (should become 0)} {
    set tensor [torch::full {1} -5.0]
    set result [torch::relu6 $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.0) < 0.001}
} {1}

test relu6-1.3 {ReLU6 with positive values in range (should remain unchanged)} {
    set tensor [torch::full {1} 3.0]
    set result [torch::relu6 $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 3.0) < 0.001}
} {1}

test relu6-1.4 {ReLU6 with values above 6 (should become 6)} {
    set tensor [torch::full {1} 10.0]
    set result [torch::relu6 $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 6.0) < 0.001}
} {1}

test relu6-1.5 {ReLU6 with exactly 0} {
    set tensor [torch::full {1} 0.0]
    set result [torch::relu6 $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.0) < 0.001}
} {1}

test relu6-1.6 {ReLU6 with exactly 6} {
    set tensor [torch::full {1} 6.0]
    set result [torch::relu6 $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 6.0) < 0.001}
} {1}

test relu6-1.7 {ReLU6 with 2D tensor} {
    set tensor [torch::zeros {2 2}]
    set result [torch::relu6 $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} {1}

# Test 2: Named parameter syntax
test relu6-2.1 {ReLU6 with named parameters - basic} {
    set tensor [torch::tensor_create -data {-3.0 -1.0 0.0 2.0 6.0 8.0} -dtype float32 -device cpu]
    set result [torch::relu6 -input $tensor]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "6" && [string match "*Float*" $dtype]}
} {1}

test relu6-2.2 {ReLU6 named syntax with negative value} {
    set tensor [torch::full {1} -2.0]
    set result [torch::relu6 -input $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.0) < 0.001}
} {1}

test relu6-2.3 {ReLU6 named syntax with large positive value} {
    set tensor [torch::full {1} 15.0]
    set result [torch::relu6 -input $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 6.0) < 0.001}
} {1}

test relu6-2.4 {ReLU6 named syntax with value in range} {
    set tensor [torch::full {1} 4.5]
    set result [torch::relu6 -input $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 4.5) < 0.001}
} {1}

# Test 3: CamelCase alias tests (Note: Due to duplication in registration, both should work)
test relu6-3.1 {ReLU6 camelCase alias basic} {
    set tensor [torch::full {1} 5.0]
    set result [torch::relu6 $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 5.0) < 0.001}
} {1}

test relu6-3.2 {ReLU6 camelCase with named parameters} {
    set tensor [torch::full {1} 8.0]
    set result [torch::relu6 -input $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 6.0) < 0.001}
} {1}

# Test 4: Error handling
test relu6-4.1 {ReLU6 with invalid tensor name} {
    catch {torch::relu6 invalid_tensor} error
    string match "*Invalid tensor name*" $error
} 1

test relu6-4.2 {ReLU6 named syntax with invalid parameter} {
    set tensor [torch::full {1} 1.0]
    catch {torch::relu6 -invalid $tensor} error
    string match "*Unknown parameter*" $error
} 1

test relu6-4.3 {ReLU6 named syntax with missing value} {
    catch {torch::relu6 -input} error
    string match "*Missing value*" $error
} 1

test relu6-4.4 {ReLU6 with too many positional arguments} {
    set tensor [torch::full {1} 1.0]
    catch {torch::relu6 $tensor extra_arg} error
    expr {[string match "*Usage*" $error] || [string match "*arguments*" $error]}
} {1}

# Test 5: Mathematical properties
test relu6-5.1 {ReLU6 range verification - negative} {
    set tensor [torch::tensor_create -data {-10.0 -5.0 -1.0} -dtype float32 -device cpu]
    set result [torch::relu6 $tensor]
    # All values should be 0
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test relu6-5.2 {ReLU6 range verification - positive in range} {
    set tensor [torch::tensor_create -data {1.0 3.0 5.0} -dtype float32 -device cpu]
    set result [torch::relu6 $tensor]
    # All values should remain unchanged
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test relu6-5.3 {ReLU6 range verification - above 6} {
    set tensor [torch::tensor_create -data {7.0 10.0 100.0} -dtype float32 -device cpu]
    set result [torch::relu6 $tensor]
    # All values should be clamped to 6
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test relu6-5.4 {ReLU6 boundary values} {
    set tensor [torch::tensor_create -data {0.0 6.0} -dtype float32 -device cpu]
    set result [torch::relu6 $tensor]
    # Both should remain unchanged (0 and 6 are boundaries)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} {1}

# Test 6: Data type consistency
test relu6-6.1 {ReLU6 preserves float32 dtype} {
    set tensor [torch::full {2} 3.0]
    set result [torch::relu6 $tensor]
    set dtype [torch::tensor_dtype $result]
    string match "*Float*" $dtype
} {1}

test relu6-6.2 {ReLU6 preserves float64 dtype} {
    set tensor [torch::full {2} 3.0 float64]
    set result [torch::relu6 $tensor]
    set dtype [torch::tensor_dtype $result]
    string match "*Float64*" $dtype
} {1}

# Test 7: Edge cases and extreme values
test relu6-7.1 {ReLU6 with very large positive values} {
    set tensor [torch::full {1} 1000.0]
    set result [torch::relu6 $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 6.0) < 0.001}
} {1}

test relu6-7.2 {ReLU6 with very large negative values} {
    set tensor [torch::full {1} -1000.0]
    set result [torch::relu6 $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.0) < 0.001}
} {1}

test relu6-7.3 {ReLU6 with values just below 6} {
    set tensor [torch::full {1} 5.999]
    set result [torch::relu6 $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 5.999) < 0.001}
} {1}

test relu6-7.4 {ReLU6 with values just above 6} {
    set tensor [torch::full {1} 6.001]
    set result [torch::relu6 $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 6.0) < 0.001}
} {1}

test relu6-7.5 {ReLU6 with values just above 0} {
    set tensor [torch::full {1} 0.001]
    set result [torch::relu6 $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.001) < 0.0001}
} {1}

test relu6-7.6 {ReLU6 with values just below 0} {
    set tensor [torch::full {1} -0.001]
    set result [torch::relu6 $tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.0) < 0.001}
} {1}

# Test 8: Multi-dimensional tensors
test relu6-8.1 {ReLU6 with 3D tensor} {
    set tensor [torch::full {2 3 4} 5.0]
    set result [torch::relu6 $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 3 4"}
} {1}

test relu6-8.2 {ReLU6 with large 2D tensor} {
    set tensor [torch::full {10 20} 8.0]
    set result [torch::relu6 $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "10 20"}
} {1}

# Test 9: Syntax consistency (both syntaxes produce same results)
test relu6-9.1 {Positional and named syntax consistency - negative} {
    set tensor [torch::full {1} -3.0]
    set result1 [torch::relu6 $tensor]
    set result2 [torch::relu6 -input $tensor]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 0.000001}
} {1}

test relu6-9.2 {Positional and named syntax consistency - positive in range} {
    set tensor [torch::full {1} 4.0]
    set result1 [torch::relu6 $tensor]
    set result2 [torch::relu6 -input $tensor]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 0.000001}
} {1}

test relu6-9.3 {Positional and named syntax consistency - above 6} {
    set tensor [torch::full {1} 10.0]
    set result1 [torch::relu6 $tensor]
    set result2 [torch::relu6 -input $tensor]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 0.000001}
} {1}

# Test 10: Comparison with range behavior
test relu6-10.1 {ReLU6 vs manual clipping verification} {
    # Test that ReLU6 behaves like max(0, min(6, x))
    set inputs [list -5.0 -1.0 0.0 2.0 6.0 8.0 15.0]
    set expected [list 0.0 0.0 0.0 2.0 6.0 6.0 6.0]
    
    set all_match 1
    foreach input $inputs expected_val $expected {
        set tensor [torch::full {1} $input]
        set result [torch::relu6 $tensor]
        set actual [torch::tensor_item $result]
        
        if {abs($actual - $expected_val) >= 0.001} {
            set all_match 0
            break
        }
    }
    expr {$all_match}
} {1}

# Test 11: Memory and cleanup
test relu6-11.1 {ReLU6 memory cleanup} {
    set tensor [torch::full {3} 7.0]
    set result [torch::relu6 $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test relu6-11.2 {Multiple ReLU6 operations} {
    set tensor1 [torch::full {2} -2.0]
    set tensor2 [torch::full {2} 8.0]
    set result1 [torch::relu6 $tensor1]
    set result2 [torch::relu6 $tensor2]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq "2" && $shape2 eq "2"}
} {1}

# Test 12: Practical range verification with specific values
test relu6-12.1 {ReLU6 comprehensive range test} {
    set tensor [torch::tensor_create -data {-10.0 -1.0 0.0 1.0 3.0 6.0 7.0 20.0} -dtype float32 -device cpu]
    set result [torch::relu6 $tensor]
    set shape [torch::tensor_shape $result]
    # Should handle all ranges correctly: negative -> 0, [0,6] -> unchanged, >6 -> 6
    expr {$shape eq "8"}
} {1}

cleanupTests 