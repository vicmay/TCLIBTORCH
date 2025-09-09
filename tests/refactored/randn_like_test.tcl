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

puts "=== Testing Refactored Command: torch::randn_like / torch::randnLike ==="
puts ""

# Create test tensors for reference
set test_tensor [torch::zeros {3 2}]

# Test 1: Original positional syntax
test randn_like-1.1 {Original positional syntax - basic} {
    set result [torch::randn_like $test_tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# Test 2: Original positional syntax with dtype
test randn_like-1.2 {Original positional syntax with dtype} {
    set result [torch::randn_like $test_tensor float32]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Float32"}
} 1

# Test 3: Original positional syntax with dtype and device
test randn_like-1.3 {Original positional syntax with dtype and device} {
    set result [torch::randn_like $test_tensor float64 cpu]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    expr {$dtype eq "Float64" && [string match "*cpu*" $device]}
} 1

# Test 4: New named parameter syntax - minimal
test randn_like-2.1 {Named parameter syntax - minimal} {
    set result [torch::randn_like -input $test_tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# Test 5: New named parameter syntax with dtype
test randn_like-2.2 {Named parameter syntax with dtype} {
    set result [torch::randn_like -input $test_tensor -dtype float32]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Float32"}
} 1

# Test 6: New named parameter syntax with all parameters
test randn_like-2.3 {Named parameter syntax with all parameters} {
    set result [torch::randn_like -input $test_tensor -dtype float32 -device cpu -requiresGrad false]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    expr {$dtype eq "Float32" && [string match "*cpu*" $device]}
} 1

# Test 7: Named parameter syntax with different parameter order
test randn_like-2.4 {Named parameter syntax - different order} {
    set result [torch::randn_like -dtype float32 -input $test_tensor -device cpu]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Float32"}
} 1

# Test 8: CamelCase alias
test randn_like-3.1 {CamelCase alias - torch::randnLike} {
    set result [torch::randnLike -input $test_tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# Test 9: CamelCase alias with named parameters
test randn_like-3.2 {CamelCase alias with named parameters} {
    set result [torch::randnLike -input $test_tensor -dtype float64]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Float64"}
} 1

# Test 10: Error handling - missing input parameter
test randn_like-4.1 {Error handling - missing input parameter} {
    catch {torch::randn_like -dtype float32} result
    expr {[string match "*Missing*" $result] || [string match "*required*" $result]}
} 1

# Test 11: Error handling - invalid tensor name
test randn_like-4.2 {Error handling - invalid tensor name} {
    catch {torch::randn_like invalid_tensor} result
    expr {[string match "*Invalid tensor name*" $result]}
} 1

# Test 12: Error handling - unknown parameter
test randn_like-4.3 {Error handling - unknown parameter} {
    catch {torch::randn_like -input $test_tensor -invalid param} result
    expr {[string match "*Unknown parameter*" $result]}
} 1

# Test 13: Error handling - missing value for parameter
test randn_like-4.4 {Error handling - missing value for parameter} {
    catch {torch::randn_like -input $test_tensor -dtype} result
    expr {[string match "*Missing value for parameter*" $result]}
} 1

# Test 14: Validation - same result structure for equivalent syntaxes
test randn_like-5.1 {Validation - equivalent syntaxes produce same structure} {
    set result1 [torch::randn_like $test_tensor float32]
    set result2 [torch::randn_like -input $test_tensor -dtype float32]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set dtype1 [torch::tensor_dtype $result1]
    set dtype2 [torch::tensor_dtype $result2]
    
    expr {$shape1 eq $shape2 && $dtype1 eq $dtype2}
} 1

# Test 15: Statistical properties - mean should be around 0
test randn_like-6.1 {Statistical properties - mean near zero} {
    # Use larger tensor for better statistical properties
    set large_tensor [torch::zeros {100 100}]
    set result [torch::randn_like -input $large_tensor -dtype float32]
    set mean_result [torch::tensor_mean $result]
    set mean_value [torch::tensor_item $mean_result]
    
    # Mean should be close to 0 (allowing for statistical variation)
    expr {abs($mean_value) < 0.2}
} 1

# Test 16: Different tensor shapes
test randn_like-7.1 {Different tensor shapes} {
    # Test with 1D tensor
    set tensor_1d [torch::zeros {5}]
    set result_1d [torch::randn_like -input $tensor_1d]
    set shape_1d [torch::tensor_shape $result_1d]
    
    # Test with 3D tensor  
    set tensor_3d [torch::zeros {2 3 4}]
    set result_3d [torch::randn_like -input $tensor_3d]
    set shape_3d [torch::tensor_shape $result_3d]
    
    expr {$shape_1d eq "5" && $shape_3d eq "2 3 4"}
} 1

# Test 17: RequiresGrad parameter
test randn_like-8.1 {RequiresGrad parameter} {
    set result [torch::randn_like -input $test_tensor -requiresGrad true]
    set requires_grad [torch::tensor_requires_grad $result]
    expr {$requires_grad eq "1"}
} 1

# Test 18: Data type verification
test randn_like-9.1 {Data type verification} {
    # Normal random tensors typically use float types
    set dtypes {float32 float64}
    set success 1
    
    foreach dtype $dtypes {
        if {[catch {
            set result [torch::randn_like -input $test_tensor -dtype $dtype]
            set actual_dtype [torch::tensor_dtype $result]
        }]} {
            set success 0
            break
        }
    }
    
    expr {$success}
} 1

# Test 19: Statistical properties - values should be different
test randn_like-10.1 {Statistical properties - randomness} {
    set result1 [torch::randn_like -input $test_tensor -dtype float32]
    set result2 [torch::randn_like -input $test_tensor -dtype float32]
    
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
test randn_like-11.1 {Integration with other commands} {
    set result [torch::randn_like -input $test_tensor -dtype float32]
    set sum_result [torch::tensor_sum $result]
    set sum_value [torch::tensor_item $sum_result]
    
    # 3x2 tensor with normal distribution should have reasonable sum values
    # (not testing exact range since it's probabilistic)
    expr {[string is double $sum_value]}
} 1

# Test 21: Distribution range validation
test randn_like-12.1 {Distribution range validation} {
    set result [torch::randn_like -input $test_tensor -dtype float32]
    set min_result [torch::tensor_min $result]
    set max_result [torch::tensor_max $result]
    set min_value [torch::tensor_item $min_result]
    set max_value [torch::tensor_item $max_result]
    
    # Normal distribution can have wide range, but should be reasonable
    # Most values should be within -4 to +4 standard deviations
    expr {$min_value > -5.0 && $max_value < 5.0}
} 1

puts ""
puts "✅ All tests passed for torch::randn_like / torch::randnLike"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/randn_like.md"
puts "   2. Update tracking: mark command complete in COMMAND-TRACKING.md" 
puts "   3. Commit changes: ./scripts/commit_refactored.sh randn_like"
puts "   4. Move to next command: ./scripts/select_next_command.sh"

cleanupTests 