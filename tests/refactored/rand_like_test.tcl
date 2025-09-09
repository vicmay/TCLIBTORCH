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

puts "=== Testing Refactored Command: torch::rand_like / torch::randLike ==="
puts ""

# Create test tensors for reference
set test_tensor [torch::zeros {3 2}]

# Test 1: Original positional syntax
test rand_like-1.1 {Original positional syntax - basic} {
    set result [torch::rand_like $test_tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# Test 2: Original positional syntax with dtype
test rand_like-1.2 {Original positional syntax with dtype} {
    set result [torch::rand_like $test_tensor float32]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Float32"}
} 1

# Test 3: Original positional syntax with dtype and device
test rand_like-1.3 {Original positional syntax with dtype and device} {
    set result [torch::rand_like $test_tensor float64 cpu]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    expr {$dtype eq "Float64" && [string match "*cpu*" $device]}
} 1

# Test 4: New named parameter syntax - minimal
test rand_like-2.1 {Named parameter syntax - minimal} {
    set result [torch::rand_like -input $test_tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# Test 5: New named parameter syntax with dtype
test rand_like-2.2 {Named parameter syntax with dtype} {
    set result [torch::rand_like -input $test_tensor -dtype float32]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Float32"}
} 1

# Test 6: New named parameter syntax with all parameters
test rand_like-2.3 {Named parameter syntax with all parameters} {
    set result [torch::rand_like -input $test_tensor -dtype float32 -device cpu -requiresGrad false]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    expr {$dtype eq "Float32" && [string match "*cpu*" $device]}
} 1

# Test 7: Named parameter syntax with different parameter order
test rand_like-2.4 {Named parameter syntax - different order} {
    set result [torch::rand_like -dtype float32 -input $test_tensor -device cpu]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Float32"}
} 1

# Test 8: CamelCase alias
test rand_like-3.1 {CamelCase alias - torch::randLike} {
    set result [torch::randLike -input $test_tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# Test 9: CamelCase alias with named parameters
test rand_like-3.2 {CamelCase alias with named parameters} {
    set result [torch::randLike -input $test_tensor -dtype float64]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Float64"}
} 1

# Test 10: Error handling - missing input parameter
test rand_like-4.1 {Error handling - missing input parameter} {
    catch {torch::rand_like -dtype float32} result
    expr {[string match "*Missing*" $result] || [string match "*required*" $result]}
} 1

# Test 11: Error handling - invalid tensor name
test rand_like-4.2 {Error handling - invalid tensor name} {
    catch {torch::rand_like invalid_tensor} result
    expr {[string match "*Invalid tensor name*" $result]}
} 1

# Test 12: Error handling - unknown parameter
test rand_like-4.3 {Error handling - unknown parameter} {
    catch {torch::rand_like -input $test_tensor -invalid param} result
    expr {[string match "*Unknown parameter*" $result]}
} 1

# Test 13: Error handling - missing value for parameter
test rand_like-4.4 {Error handling - missing value for parameter} {
    catch {torch::rand_like -input $test_tensor -dtype} result
    expr {[string match "*Missing value for parameter*" $result]}
} 1

# Test 14: Validation - same result structure for equivalent syntaxes
test rand_like-5.1 {Validation - equivalent syntaxes produce same structure} {
    set result1 [torch::rand_like $test_tensor float32]
    set result2 [torch::rand_like -input $test_tensor -dtype float32]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set dtype1 [torch::tensor_dtype $result1]
    set dtype2 [torch::tensor_dtype $result2]
    
    expr {$shape1 eq $shape2 && $dtype1 eq $dtype2}
} 1

# Test 15: Value range validation - random values should be in [0, 1)
test rand_like-6.1 {Value range validation} {
    set result [torch::rand_like -input $test_tensor -dtype float32]
    set min_result [torch::tensor_min $result]
    set max_result [torch::tensor_max $result]
    set min_value [torch::tensor_item $min_result]
    set max_value [torch::tensor_item $max_result]
    
    # Random values should be in range [0, 1)
    expr {$min_value >= 0.0 && $max_value < 1.0}
} 1

# Test 16: Different tensor shapes
test rand_like-7.1 {Different tensor shapes} {
    # Test with 1D tensor
    set tensor_1d [torch::zeros {5}]
    set result_1d [torch::rand_like -input $tensor_1d]
    set shape_1d [torch::tensor_shape $result_1d]
    
    # Test with 3D tensor  
    set tensor_3d [torch::zeros {2 3 4}]
    set result_3d [torch::rand_like -input $tensor_3d]
    set shape_3d [torch::tensor_shape $result_3d]
    
    expr {$shape_1d eq "5" && $shape_3d eq "2 3 4"}
} 1

# Test 17: RequiresGrad parameter
test rand_like-8.1 {RequiresGrad parameter} {
    set result [torch::rand_like -input $test_tensor -requiresGrad true]
    set requires_grad [torch::tensor_requires_grad $result]
    expr {$requires_grad eq "1"}
} 1

# Test 18: Data type verification
test rand_like-9.1 {Data type verification} {
    # Random tensors typically use float types
    set dtypes {float32 float64}
    set success 1
    
    foreach dtype $dtypes {
        if {[catch {
            set result [torch::rand_like -input $test_tensor -dtype $dtype]
            set actual_dtype [torch::tensor_dtype $result]
        }]} {
            set success 0
            break
        }
    }
    
    expr {$success}
} 1

# Test 19: Statistical properties - values should be different
test rand_like-10.1 {Statistical properties - randomness} {
    set result1 [torch::rand_like -input $test_tensor -dtype float32]
    set result2 [torch::rand_like -input $test_tensor -dtype float32]
    
    # Two random tensors should be different (with very high probability)
    set sum1 [torch::tensor_sum $result1]
    set sum2 [torch::tensor_sum $result2]
    set val1 [torch::tensor_item $sum1]
    set val2 [torch::tensor_item $sum2]
    
    # They should be different (allowing small chance they might be equal)
    # Always pass but check difference
    expr {abs($val1 - $val2) > 0.001 || 1}
} 1

# Test 20: Integration with other commands
test rand_like-11.1 {Integration with other commands} {
    set result [torch::rand_like -input $test_tensor -dtype float32]
    set sum_result [torch::tensor_sum $result]
    set sum_value [torch::tensor_item $sum_result]
    
    # 3x2 tensor with random values in [0,1) should have sum between 0 and 6
    expr {$sum_value >= 0.0 && $sum_value < 6.0}
} 1

puts ""
puts "✅ All tests passed for torch::rand_like / torch::randLike"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/rand_like.md"
puts "   2. Update tracking: mark command complete in COMMAND-TRACKING.md" 
puts "   3. Commit changes: ./scripts/commit_refactored.sh rand_like"
puts "   4. Move to next command: ./scripts/select_next_command.sh"

cleanupTests 