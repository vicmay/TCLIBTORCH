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

# Test 1: Basic functionality with positional syntax (default parameters)
test hardtanh-1.1 {Basic hardtanh with positional syntax - default range} {
    set tensor [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32 -device cpu]
    set result [torch::hardtanh $tensor]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "5" && [string match "*Float*" $dtype]}
} {1}

test hardtanh-1.2 {HardTanh with zero tensor} {
    set tensor [torch::zeros {3}]
    set result [torch::hardtanh $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test hardtanh-1.3 {HardTanh with custom min/max values} {
    set tensor [torch::full {3} 2.0]
    set result [torch::hardtanh $tensor -0.5 0.5]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test hardtanh-1.4 {HardTanh with negative values} {
    set tensor [torch::full {3} -1.5]
    set result [torch::hardtanh $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test hardtanh-1.5 {HardTanh with 2D tensor} {
    set tensor [torch::zeros {2 2}]
    set result [torch::hardtanh $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} {1}

test hardtanh-1.6 {HardTanh with single value - clipping test} {
    set tensor [torch::full {1} 2.0]
    set result [torch::hardtanh $tensor]
    set value [torch::tensor_item $result]
    # Hard tanh clips to [-1, 1] by default, so 2.0 should become 1.0
    expr {abs($value - 1.0) < 0.01}
} {1}

test hardtanh-1.7 {HardTanh with value below range} {
    set tensor [torch::full {1} -2.0]
    set result [torch::hardtanh $tensor]
    set value [torch::tensor_item $result]
    # Hard tanh clips to [-1, 1] by default, so -2.0 should become -1.0
    expr {abs($value - (-1.0)) < 0.01}
} {1}

test hardtanh-1.8 {HardTanh with value in range} {
    set tensor [torch::full {1} 0.5]
    set result [torch::hardtanh $tensor]
    set value [torch::tensor_item $result]
    # Value in range should remain unchanged
    expr {abs($value - 0.5) < 0.01}
} {1}

# Test 2: Named parameter syntax
test hardtanh-2.1 {HardTanh with named parameters - basic} {
    set tensor [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32 -device cpu]
    set result [torch::hardtanh -input $tensor]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "5" && [string match "*Float*" $dtype]}
} {1}

test hardtanh-2.2 {HardTanh with named parameters - custom range} {
    set tensor [torch::full {1} 2.0]
    set result [torch::hardtanh -input $tensor -min -0.5 -max 0.5]
    set value [torch::tensor_item $result]
    # Should clip 2.0 to 0.5
    expr {abs($value - 0.5) < 0.01}
} {1}

test hardtanh-2.3 {HardTanh with named parameters - alternative parameter names} {
    set tensor [torch::full {1} -2.0]
    set result [torch::hardtanh -input $tensor -minVal -0.5 -maxVal 0.5]
    set value [torch::tensor_item $result]
    # Should clip -2.0 to -0.5
    expr {abs($value - (-0.5)) < 0.01}
} {1}

test hardtanh-2.4 {HardTanh named syntax with 2D tensor} {
    set tensor [torch::ones {2 2}]
    set result [torch::hardtanh -input $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} {1}

test hardtanh-2.5 {HardTanh named syntax with zeros} {
    set tensor [torch::zeros {4}]
    set result [torch::hardtanh -input $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "4"}
} {1}

# Test 3: CamelCase alias tests
test hardtanh-3.1 {HardTanh camelCase alias basic} {
    set tensor [torch::full {3} 1.0]
    set result [torch::hardTanh $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test hardtanh-3.2 {HardTanh camelCase with named parameters} {
    set tensor [torch::full {1} 3.0]
    set result [torch::hardTanh -input $tensor -min -2.0 -max 2.0]
    set value [torch::tensor_item $result]
    # Should clip 3.0 to 2.0
    expr {abs($value - 2.0) < 0.01}
} {1}

test hardtanh-3.3 {HardTanh camelCase with positional parameters} {
    set tensor [torch::full {1} 3.0]
    set result [torch::hardTanh $tensor -2.0 2.0]
    set value [torch::tensor_item $result]
    # Should clip 3.0 to 2.0
    expr {abs($value - 2.0) < 0.01}
} {1}

# Test 4: Error handling
test hardtanh-4.1 {HardTanh with invalid tensor name} {
    catch {torch::hardtanh invalid_tensor} error
    string match "*Invalid tensor name*" $error
} 1

test hardtanh-4.2 {HardTanh with missing arguments} {
    set result [catch {torch::hardtanh} error]
    expr {$result == 0 || $result == 1}
} {1}

test hardtanh-4.3 {HardTanh with too many arguments} {
    set tensor [torch::full {1} 1.0]
    catch {torch::hardtanh $tensor 1.0 2.0 3.0} error
    expr {[string match "*Usage*" $error] || [string match "*arguments*" $error]}
} {1}

test hardtanh-4.4 {HardTanh named syntax with invalid parameter} {
    set tensor [torch::full {1} 1.0]
    catch {torch::hardtanh -invalid $tensor} error
    string match "*Unknown parameter*" $error
} 1

test hardtanh-4.5 {HardTanh named syntax with missing value} {
    catch {torch::hardtanh -input} error
    string match "*Missing value*" $error
} 1

test hardtanh-4.6 {HardTanh with invalid min/max order} {
    set tensor [torch::full {1} 1.0]
    catch {torch::hardtanh -input $tensor -min 1.0 -max -1.0} error
    string match "*min_val must be <= max_val*" $error
} 1

test hardtanh-4.7 {HardTanh with invalid min value} {
    set tensor [torch::full {1} 1.0]
    catch {torch::hardtanh $tensor invalid_min} error
    string match "*Invalid min_val*" $error
} 1

test hardtanh-4.8 {HardTanh with invalid max value} {
    set tensor [torch::full {1} 1.0]
    catch {torch::hardtanh $tensor -1.0 invalid_max} error
    string match "*Invalid max_val*" $error
} 1

# Test 5: Data type consistency
test hardtanh-5.1 {HardTanh preserves float32 dtype} {
    set tensor [torch::full {2} 1.0]
    set result [torch::hardtanh $tensor]
    set dtype [torch::tensor_dtype $result]
    string match "*Float*" $dtype
} {1}

test hardtanh-5.2 {HardTanh preserves float64 dtype} {
    set tensor [torch::full {2} 1.0 float64]
    set result [torch::hardtanh $tensor]
    set dtype [torch::tensor_dtype $result]
    string match "*Float64*" $dtype
} {1}

# Test 6: Mathematical properties
test hardtanh-6.1 {HardTanh mathematical correctness - zero input} {
    set tensor [torch::full {1} 0.0]
    set result [torch::hardtanh $tensor]
    set value [torch::tensor_item $result]
    # Hard tanh of 0 should remain 0
    expr {abs($value) < 0.01}
} {1}

test hardtanh-6.2 {HardTanh clipping behavior} {
    set tensor [torch::tensor_create -data {-3.0 -1.0 0.0 1.0 3.0} -dtype float32 -device cpu]
    set result [torch::hardtanh $tensor]
    set shape [torch::tensor_shape $result]
    # All values should be clipped to [-1, 1] range
    expr {$shape eq "5"}
} {1}

test hardtanh-6.3 {HardTanh with custom range} {
    set tensor [torch::full {1} 5.0]
    set result [torch::hardtanh $tensor -2.0 2.0]
    set value [torch::tensor_item $result]
    # Should clip 5.0 to 2.0
    expr {abs($value - 2.0) < 0.01}
} {1}

test hardtanh-6.4 {HardTanh symmetric behavior} {
    set tensor_pos [torch::full {1} 2.0]
    set tensor_neg [torch::full {1} -2.0]
    set result_pos [torch::hardtanh $tensor_pos]
    set result_neg [torch::hardtanh $tensor_neg]
    set value_pos [torch::tensor_item $result_pos]
    set value_neg [torch::tensor_item $result_neg]
    # Should be symmetric: hardtanh(2) = 1, hardtanh(-2) = -1
    expr {abs($value_pos - 1.0) < 0.01 && abs($value_neg - (-1.0)) < 0.01}
} {1}

# Test 7: Edge cases
test hardtanh-7.1 {HardTanh with very large positive values} {
    set tensor [torch::full {2} 100.0]
    set result [torch::hardtanh $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} {1}

test hardtanh-7.2 {HardTanh with very large negative values} {
    set tensor [torch::full {2} -100.0]
    set result [torch::hardtanh $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} {1}

test hardtanh-7.3 {HardTanh with zero range (min == max)} {
    set tensor [torch::full {1} 5.0]
    set result [torch::hardtanh $tensor 0.0 0.0]
    set value [torch::tensor_item $result]
    # Should clip everything to 0.0
    expr {abs($value) < 0.01}
} {1}

test hardtanh-7.4 {HardTanh with very small range} {
    set tensor [torch::full {1} 1.0]
    set result [torch::hardtanh $tensor 0.1 0.2]
    set value [torch::tensor_item $result]
    # Should clip 1.0 to 0.2
    expr {abs($value - 0.2) < 0.01}
} {1}

# Test 8: Multi-dimensional tensors
test hardtanh-8.1 {HardTanh with 3D tensor} {
    set tensor [torch::full {2 3 4} 2.0]
    set result [torch::hardtanh $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 3 4"}
} {1}

test hardtanh-8.2 {HardTanh with large 2D tensor} {
    set tensor [torch::full {10 20} -2.0]
    set result [torch::hardtanh $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "10 20"}
} {1}

# Test 9: Syntax consistency (both syntaxes produce same results)
test hardtanh-9.1 {Positional and named syntax consistency} {
    set tensor [torch::full {1} 2.0]
    set result1 [torch::hardtanh $tensor]
    set result2 [torch::hardtanh -input $tensor]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 0.000001}
} {1}

test hardtanh-9.2 {Snake_case and camelCase consistency} {
    set tensor [torch::full {1} 1.5]
    set result1 [torch::hardtanh $tensor]
    set result2 [torch::hardTanh $tensor]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 0.000001}
} {1}

test hardtanh-9.3 {All syntax variations consistency with custom range} {
    set tensor [torch::full {1} 3.0]
    set result1 [torch::hardtanh $tensor -0.5 0.5]
    set result2 [torch::hardtanh -input $tensor -min -0.5 -max 0.5]
    set result3 [torch::hardTanh $tensor -0.5 0.5]
    set result4 [torch::hardTanh -input $tensor -min -0.5 -max 0.5]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    set value3 [torch::tensor_item $result3]
    set value4 [torch::tensor_item $result4]
    
    expr {abs($value1 - $value2) < 0.000001 && abs($value1 - $value3) < 0.000001 && abs($value1 - $value4) < 0.000001}
} {1}

# Test 10: Parameter validation
test hardtanh-10.1 {Parameter order independence in named syntax} {
    set tensor [torch::full {1} 2.0]
    set result1 [torch::hardtanh -input $tensor -min -0.5 -max 0.5]
    set result2 [torch::hardtanh -min -0.5 -max 0.5 -input $tensor]
    set result3 [torch::hardtanh -max 0.5 -input $tensor -min -0.5]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    set value3 [torch::tensor_item $result3]
    
    expr {abs($value1 - $value2) < 0.000001 && abs($value1 - $value3) < 0.000001}
} {1}

test hardtanh-10.2 {Mixed parameter names consistency} {
    set tensor [torch::full {1} 2.0]
    set result1 [torch::hardtanh -input $tensor -min -0.5 -max 0.5]
    set result2 [torch::hardtanh -input $tensor -minVal -0.5 -maxVal 0.5]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 0.000001}
} {1}

# Test 11: Memory and cleanup
test hardtanh-11.1 {HardTanh memory cleanup} {
    set tensor [torch::full {3} 1.0]
    set result [torch::hardtanh $tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} {1}

test hardtanh-11.2 {Multiple HardTanh operations} {
    set tensor1 [torch::full {2} -2.0]
    set tensor2 [torch::full {2} 2.0]
    set result1 [torch::hardtanh $tensor1]
    set result2 [torch::hardtanh $tensor2]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq "2" && $shape2 eq "2"}
} {1}

cleanupTests 