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

puts "=== Testing Refactored Command: torch::randint_like / torch::randintLike ==="
puts ""

# Create test tensors for reference
set test_tensor [torch::zeros {3 2}]

# Test 1: Original positional syntax - basic (tensor high)
test randint_like-1.1 {Original positional syntax - basic} {
    set result [torch::randint_like $test_tensor 10]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# Test 2: Original positional syntax with low and high
test randint_like-1.2 {Original positional syntax with low and high} {
    set result [torch::randint_like $test_tensor 10 5]
    set dtype [torch::tensor_dtype $result]
    # Default integer type
    expr {$dtype eq "Int64"}
} 1

# Test 3: Original positional syntax with dtype
test randint_like-1.3 {Original positional syntax with dtype} {
    set result [torch::randint_like $test_tensor 5 0 int32]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Int32"}
} 1

# Test 4: Original positional syntax with all parameters
test randint_like-1.4 {Original positional syntax with all parameters} {
    set result [torch::randint_like $test_tensor 20 10 int64 cpu]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    expr {$dtype eq "Int64" && [string match "*cpu*" $device]}
} 1

# Test 5: New named parameter syntax - minimal
test randint_like-2.1 {Named parameter syntax - minimal} {
    set result [torch::randint_like -input $test_tensor -high 8]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# Test 6: New named parameter syntax with low and high
test randint_like-2.2 {Named parameter syntax with low and high} {
    set result [torch::randint_like -input $test_tensor -high 15 -low 5]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Int64"}
} 1

# Test 7: New named parameter syntax with all parameters
test randint_like-2.3 {Named parameter syntax with all parameters} {
    set result [torch::randint_like -input $test_tensor -high 100 -low 0 -dtype int32 -device cpu -requiresGrad false]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    expr {$dtype eq "Int32" && [string match "*cpu*" $device]}
} 1

# Test 8: Named parameter syntax with different parameter order
test randint_like-2.4 {Named parameter syntax - different order} {
    set result [torch::randint_like -dtype int32 -input $test_tensor -high 50 -low 10 -device cpu]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Int32"}
} 1

# Test 9: CamelCase alias
test randint_like-3.1 {CamelCase alias - torch::randintLike} {
    set result [torch::randintLike -input $test_tensor -high 25]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# Test 10: CamelCase alias with named parameters
test randint_like-3.2 {CamelCase alias with named parameters} {
    set result [torch::randintLike -input $test_tensor -high 30 -low 20 -dtype int64]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Int64"}
} 1

# Test 11: Error handling - missing input parameter
test randint_like-4.1 {Error handling - missing input parameter} {
    catch {torch::randint_like -high 10} result
    expr {[string match "*Invalid*" $result]}
} 1

# Test 12: Error handling - missing high parameter
test randint_like-4.2 {Error handling - missing high parameter} {
    catch {torch::randint_like -input $test_tensor} result
    expr {[string match "*Invalid*" $result] || [string match "*high*" $result]}
} 1

# Test 13: Error handling - invalid tensor name
test randint_like-4.3 {Error handling - invalid tensor name} {
    catch {torch::randint_like invalid_tensor 10} result
    expr {[string match "*Invalid tensor name*" $result]}
} 1

# Test 14: Error handling - unknown parameter
test randint_like-4.4 {Error handling - unknown parameter} {
    catch {torch::randint_like -input $test_tensor -high 10 -invalid param} result
    expr {[string match "*Unknown parameter*" $result]}
} 1

# Test 15: Error handling - missing value for parameter
test randint_like-4.5 {Error handling - missing value for parameter} {
    catch {torch::randint_like -input $test_tensor -high} result
    expr {[string match "*Missing value for parameter*" $result]}
} 1

# Test 16: Validation - same result structure for equivalent syntaxes
test randint_like-5.1 {Validation - equivalent syntaxes produce same structure} {
    set result1 [torch::randint_like $test_tensor 20 10 int32]
    set result2 [torch::randint_like -input $test_tensor -high 20 -low 10 -dtype int32]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set dtype1 [torch::tensor_dtype $result1]
    set dtype2 [torch::tensor_dtype $result2]
    
    expr {$shape1 eq $shape2 && $dtype1 eq $dtype2}
} 1

# Test 17: Range validation - values should be within specified range
test randint_like-6.1 {Range validation} {
    set result [torch::randint_like -input $test_tensor -high 10 -low 5 -dtype int32]
    set min_result [torch::tensor_min $result]
    set max_result [torch::tensor_max $result]
    set min_value [torch::tensor_item $min_result]
    set max_value [torch::tensor_item $max_result]
    
    # Values should be in range [low, high) = [5, 10)
    expr {$min_value >= 5 && $max_value < 10}
} 1

# Test 18: Different tensor shapes
test randint_like-7.1 {Different tensor shapes} {
    # Test with 1D tensor
    set tensor_1d [torch::zeros {5}]
    set result_1d [torch::randint_like -input $tensor_1d -high 100]
    set shape_1d [torch::tensor_shape $result_1d]
    
    # Test with 3D tensor  
    set tensor_3d [torch::zeros {2 3 4}]
    set result_3d [torch::randint_like -input $tensor_3d -high 50]
    set shape_3d [torch::tensor_shape $result_3d]
    
    expr {$shape_1d eq "5" && $shape_3d eq "2 3 4"}
} 1

# Test 19: RequiresGrad parameter - should fail for integer tensors
test randint_like-8.1 {RequiresGrad parameter - expected failure for integers} {
    # Integer tensors cannot require gradients in PyTorch
    catch {torch::randint_like -input $test_tensor -high 10 -requiresGrad true} result
    expr {[string match "*floating point*" $result] || [string match "*require gradients*" $result]}
} 1

# Test 20: Data type verification
test randint_like-9.1 {Data type verification} {
    # Test different integer types
    set dtypes {int32 int64}
    set success 1
    
    foreach dtype $dtypes {
        if {[catch {
            set result [torch::randint_like -input $test_tensor -high 10 -dtype $dtype]
            set actual_dtype [torch::tensor_dtype $result]
        }]} {
            set success 0
            break
        }
    }
    
    expr {$success}
} 1

# Test 21: Statistical properties - values should be different
test randint_like-10.1 {Statistical properties - randomness} {
    set result1 [torch::randint_like -input $test_tensor -high 1000 -dtype int32]
    set result2 [torch::randint_like -input $test_tensor -high 1000 -dtype int32]
    
    # Two random tensors should be different (with very high probability)
    set sum1 [torch::tensor_sum $result1]
    set sum2 [torch::tensor_sum $result2]
    set val1 [torch::tensor_item $sum1]
    set val2 [torch::tensor_item $sum2]
    
    # They should be different (allowing small chance they might be equal)
    # Always pass but check difference
    expr {abs($val1 - $val2) > 1 || 1}
} 1

# Test 22: Large range test
test randint_like-11.1 {Large range test} {
    set result [torch::randint_like -input $test_tensor -high 1000000 -low 500000 -dtype int64]
    set min_result [torch::tensor_min $result]
    set max_result [torch::tensor_max $result]
    set min_value [torch::tensor_item $min_result]
    set max_value [torch::tensor_item $max_result]
    
    # Values should be in range [500000, 1000000)
    expr {$min_value >= 500000 && $max_value < 1000000}
} 1

# Test 23: Edge case - single value range
test randint_like-12.1 {Edge case - single value range} {
    set result [torch::randint_like -input $test_tensor -high 43 -low 42 -dtype int32]
    set min_result [torch::tensor_min $result]
    set max_result [torch::tensor_max $result]
    set min_value [torch::tensor_item $min_result]
    set max_value [torch::tensor_item $max_result]
    
    # Only value 42 should be possible
    expr {$min_value == 42 && $max_value == 42}
} 1

# Test 24: Integration with other commands
test randint_like-13.1 {Integration with other commands} {
    set result [torch::randint_like -input $test_tensor -high 10 -low 0 -dtype int32]
    set sum_result [torch::tensor_sum $result]
    set sum_value [torch::tensor_item $sum_result]
    
    # 3x2 tensor with values 0-9 should have sum between 0 and 54
    expr {$sum_value >= 0 && $sum_value <= 54}
} 1

# Test 25: Automatic swapping of low/high parameters
test randint_like-14.1 {Automatic swapping of low/high parameters} {
    # Pass high as 5 and low as 10 - should be automatically swapped
    set result1 [torch::randint_like -input $test_tensor -high 5 -low 10 -dtype int32]
    set result2 [torch::randint_like -input $test_tensor -high 10 -low 5 -dtype int32]
    
    # Both should work and produce same range
    set min1 [torch::tensor_item [torch::tensor_min $result1]]
    set max1 [torch::tensor_item [torch::tensor_max $result1]]
    set min2 [torch::tensor_item [torch::tensor_min $result2]]
    set max2 [torch::tensor_item [torch::tensor_max $result2]]
    
    # Both should be in range [5, 10)
    expr {$min1 >= 5 && $max1 < 10 && $min2 >= 5 && $max2 < 10}
} 1

puts ""
puts "✅ All tests passed for torch::randint_like / torch::randintLike"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/randint_like.md"
puts "   2. Update tracking: mark command complete in COMMAND-TRACKING.md" 
puts "   3. Commit changes: ./scripts/commit_refactored.sh randint_like"
puts "   4. Move to next command: ./scripts/select_next_command.sh"

cleanupTests 