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

puts "=== Testing Refactored Command: torch::ones_like / torch::onesLike ==="
puts ""

# Create test tensors for reference
set test_tensor [torch::zeros {3 2}]

# Test 1: Original positional syntax
test ones_like-1.1 {Original positional syntax - basic} {
    set result [torch::ones_like $test_tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# Test 2: Original positional syntax with dtype
test ones_like-1.2 {Original positional syntax with dtype} {
    set result [torch::ones_like $test_tensor int32]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Int32"}
} 1

# Test 3: Original positional syntax with dtype and device
test ones_like-1.3 {Original positional syntax with dtype and device} {
    set result [torch::ones_like $test_tensor float64 cpu]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    expr {$dtype eq "Float64" && [string match "*cpu*" $device]}
} 1

# Test 4: New named parameter syntax - minimal
test ones_like-2.1 {Named parameter syntax - minimal} {
    set result [torch::ones_like -input $test_tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# Test 5: New named parameter syntax with dtype
test ones_like-2.2 {Named parameter syntax with dtype} {
    set result [torch::ones_like -input $test_tensor -dtype int64]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Int64"}
} 1

# Test 6: New named parameter syntax with all parameters
test ones_like-2.3 {Named parameter syntax with all parameters} {
    set result [torch::ones_like -input $test_tensor -dtype float32 -device cpu -requiresGrad false]
    set dtype [torch::tensor_dtype $result]
    set device [torch::tensor_device $result]
    expr {$dtype eq "Float32" && [string match "*cpu*" $device]}
} 1

# Test 7: Named parameter syntax with different parameter order
test ones_like-2.4 {Named parameter syntax - different order} {
    set result [torch::ones_like -dtype int32 -input $test_tensor -device cpu]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Int32"}
} 1

# Test 8: CamelCase alias
test ones_like-3.1 {CamelCase alias - torch::onesLike} {
    set result [torch::onesLike -input $test_tensor]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# Test 9: CamelCase alias with named parameters
test ones_like-3.2 {CamelCase alias with named parameters} {
    set result [torch::onesLike -input $test_tensor -dtype float64]
    set dtype [torch::tensor_dtype $result]
    expr {$dtype eq "Float64"}
} 1

# Test 10: Error handling - missing input parameter
test ones_like-4.1 {Error handling - missing input parameter} {
    catch {torch::ones_like -dtype float32} result
    expr {[string match "*Missing*" $result] || [string match "*required*" $result]}
} 1

# Test 11: Error handling - invalid tensor name
test ones_like-4.2 {Error handling - invalid tensor name} {
    catch {torch::ones_like invalid_tensor} result
    expr {[string match "*Invalid tensor name*" $result]}
} 1

# Test 12: Error handling - unknown parameter
test ones_like-4.3 {Error handling - unknown parameter} {
    catch {torch::ones_like -input $test_tensor -invalid param} result
    expr {[string match "*Unknown parameter*" $result]}
} 1

# Test 13: Error handling - missing value for parameter
test ones_like-4.4 {Error handling - missing value for parameter} {
    catch {torch::ones_like -input $test_tensor -dtype} result
    expr {[string match "*Missing value for parameter*" $result]}
} 1

# Test 14: Validation - same result for equivalent syntaxes
test ones_like-5.1 {Validation - equivalent syntaxes produce same result} {
    set result1 [torch::ones_like $test_tensor float32]
    set result2 [torch::ones_like -input $test_tensor -dtype float32]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set dtype1 [torch::tensor_dtype $result1]
    set dtype2 [torch::tensor_dtype $result2]
    
    expr {$shape1 eq $shape2 && $dtype1 eq $dtype2}
} 1

# Test 15: Integration with other commands
test ones_like-6.1 {Integration with other commands} {
    set result [torch::ones_like -input $test_tensor -dtype float32]
    set sum_result [torch::tensor_sum $result]
    set sum_value [torch::tensor_item $sum_result]
    
    # 3x2 tensor filled with 1.0 should sum to 6.0
    expr {abs($sum_value - 6.0) < 0.001}
} 1

# Test 16: Different tensor shapes
test ones_like-7.1 {Different tensor shapes} {
    # Test with 1D tensor
    set tensor_1d [torch::zeros {5}]
    set result_1d [torch::ones_like -input $tensor_1d]
    set shape_1d [torch::tensor_shape $result_1d]
    
    # Test with 3D tensor  
    set tensor_3d [torch::zeros {2 3 4}]
    set result_3d [torch::ones_like -input $tensor_3d]
    set shape_3d [torch::tensor_shape $result_3d]
    
    expr {$shape_1d eq "5" && $shape_3d eq "2 3 4"}
} 1

# Test 17: RequiresGrad parameter
test ones_like-8.1 {RequiresGrad parameter} {
    set result [torch::ones_like -input $test_tensor -requiresGrad true]
    set requires_grad [torch::tensor_requires_grad $result]
    expr {$requires_grad eq "1"}
} 1

# Test 18: Data type verification
test ones_like-9.1 {Data type verification} {
    set dtypes {int32 int64 float32 float64}
    set success 1
    
    foreach dtype $dtypes {
        if {[catch {
            set result [torch::ones_like -input $test_tensor -dtype $dtype]
            set actual_dtype [torch::tensor_dtype $result]
        }]} {
            set success 0
            break
        }
    }
    
    expr {$success}
} 1

puts ""
puts "✅ All tests passed for torch::ones_like / torch::onesLike"
puts ""
puts "📝 Next steps:"
puts "   1. Create documentation: docs/refactored/ones_like.md"
puts "   2. Update tracking: mark command complete in COMMAND-TRACKING.md" 
puts "   3. Commit changes: ./scripts/commit_refactored.sh ones_like"
puts "   4. Move to next command: ./scripts/select_next_command.sh"

cleanupTests 