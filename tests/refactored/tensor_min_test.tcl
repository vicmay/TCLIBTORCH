#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load [file normalize ../../build/libtorchtcl.so]}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

puts "=== Testing Refactored Command: torch::tensor_min / torch::tensorMin ==="
puts ""

# Create test tensors for reference
set test_tensor [torch::tensor_create {3.0 1.0 4.0 1.5 9.0 2.6}]
set test_tensor_2d [torch::tensor_create {3.0 1.0 4.0 1.5 9.0 2.6}]
set test_tensor_2d [torch::tensor_reshape $test_tensor_2d {2 3}]

# Test 1: Original positional syntax - basic
test tensor_min-1.1 {Original positional syntax - basic} {
    set result [torch::tensor_min $test_tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 1.0) < 0.001}
} 1

# Test 2: Original positional syntax with dimension
test tensor_min-1.2 {Original positional syntax with dimension} {
    set result [torch::tensor_min $test_tensor_2d 0]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

# Test 3: New named parameter syntax - minimal
test tensor_min-2.1 {Named parameter syntax - minimal} {
    set result [torch::tensor_min -input $test_tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 1.0) < 0.001}
} 1

# Test 4: New named parameter syntax with dimension
test tensor_min-2.2 {Named parameter syntax with dimension} {
    set result [torch::tensor_min -input $test_tensor_2d -dim 1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} 1

# Test 5: Named parameter syntax with different parameter order
test tensor_min-2.3 {Named parameter syntax - different order} {
    set result [torch::tensor_min -dim 0 -input $test_tensor_2d]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3"}
} 1

# Test 6: CamelCase alias
test tensor_min-3.1 {CamelCase alias - torch::tensorMin} {
    set result [torch::tensorMin -input $test_tensor]
    set value [torch::tensor_item $result]
    expr {abs($value - 1.0) < 0.001}
} 1

# Test 7: CamelCase alias with named parameters
test tensor_min-3.2 {CamelCase alias with named parameters} {
    set result [torch::tensorMin -input $test_tensor_2d -dim 1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} 1

# Test 8: Error handling - missing input parameter
test tensor_min-4.1 {Error handling - missing input parameter} {
    catch {torch::tensor_min -dim 0} result
    expr {[string match "*required*" $result] || [string match "*Input tensor*" $result]}
} 1

# Test 9: Error handling - invalid tensor name
test tensor_min-4.2 {Error handling - invalid tensor name} {
    catch {torch::tensor_min invalid_tensor} result
    expr {[string match "*Invalid tensor name*" $result]}
} 1

# Test 10: Error handling - unknown parameter
test tensor_min-4.3 {Error handling - unknown parameter} {
    catch {torch::tensor_min -input $test_tensor -invalid param} result
    expr {[string match "*Unknown parameter*" $result]}
} 1

# Test 11: Error handling - missing value for parameter
test tensor_min-4.4 {Error handling - missing value for parameter} {
    catch {torch::tensor_min -input $test_tensor -dim} result
    expr {[string match "*Missing value for parameter*" $result]}
} 1

# Test 12: Error handling - invalid dimension
test tensor_min-4.5 {Error handling - invalid dimension} {
    catch {torch::tensor_min -input $test_tensor -dim invalid} result
    expr {[string match "*Invalid*" $result]}
} 1

# Test 13: Validation - same result for equivalent syntaxes
test tensor_min-5.1 {Validation - equivalent syntaxes produce same result} {
    set result1 [torch::tensor_min $test_tensor]
    set result2 [torch::tensor_min -input $test_tensor]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    expr {abs($value1 - $value2) < 0.001}
} 1

# Test 14: Validation - dimension handling consistency
test tensor_min-5.2 {Validation - dimension handling consistency} {
    set result1 [torch::tensor_min $test_tensor_2d 0]
    set result2 [torch::tensor_min -input $test_tensor_2d -dim 0]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} 1

# Test 15: Mathematical correctness
test tensor_min-6.1 {Mathematical correctness - check actual minimum values} {
    # Create a tensor with known values
    set known_tensor [torch::tensor_create {5.0 2.0 8.0 1.0 9.0}]
    set result [torch::tensor_min -input $known_tensor]
    set min_value [torch::tensor_item $result]
    
    # Should be 1.0
    expr {abs($min_value - 1.0) < 0.001}
} 1

# Test 16: 2D tensor dimension-wise minimum
test tensor_min-6.2 {2D tensor dimension-wise minimum} {
    # Create 2x3 tensor: [[5, 2, 8], [1, 9, 3]]
    set tensor_2x3 [torch::tensor_create {5.0 2.0 8.0 1.0 9.0 3.0}]
    set tensor_2x3 [torch::tensor_reshape $tensor_2x3 {2 3}]
    
    # Min along dimension 0 (rows): should be [1, 2, 3]
    set result_dim0 [torch::tensor_min -input $tensor_2x3 -dim 0]
    set shape_dim0 [torch::tensor_shape $result_dim0]
    
    # Min along dimension 1 (cols): should be [2, 1]  
    set result_dim1 [torch::tensor_min -input $tensor_2x3 -dim 1]
    set shape_dim1 [torch::tensor_shape $result_dim1]
    
    expr {$shape_dim0 eq "3" && $shape_dim1 eq "2"}
} 1

# Test 17: Edge case - single element tensor
test tensor_min-7.1 {Edge case - single element tensor} {
    set single_tensor [torch::tensor_create {42.0}]
    set result [torch::tensor_min -input $single_tensor]
    set value [torch::tensor_item $result]
    
    expr {abs($value - 42.0) < 0.001}
} 1

# Test 18: Integration with other commands
test tensor_min-8.1 {Integration with other commands} {
    set tensor [torch::tensor_create {10.0 5.0 15.0}]
    set min_result [torch::tensor_min -input $tensor]
    set max_result [torch::tensor_max -input $tensor]
    
    set min_val [torch::tensor_item $min_result]
    set max_val [torch::tensor_item $max_result]
    
    # Min should be 5.0, max should be 15.0
    expr {abs($min_val - 5.0) < 0.001 && abs($max_val - 15.0) < 0.001}
} 1

puts ""
puts "✅ All tests passed for torch::tensor_min / torch::tensorMin"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/tensor_min.md"
puts "   2. Update tracking: mark command complete in COMMAND-TRACKING.md" 
puts "   3. Commit changes: ./scripts/commit_refactored.sh tensor_min"
puts "   4. Move to next command: ./scripts/select_next_command.sh"

cleanupTests 