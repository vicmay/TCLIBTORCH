#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic NLL loss with positional syntax
test nll_loss-1.1 {Basic NLL loss with positional syntax} {
    # Create predicted log probabilities for 2 samples, 3 classes
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -2.0 -1.0 -3.0} -shape {2 3} -dtype float32 -device cpu -requiresGrad false]
    
    # Create target tensor (class indices) - 2 samples
    set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]
    
    # Compute NLL loss using positional syntax
    set loss [torch::nll_loss $input $target]
    
    # Verify result
    set result_value [torch::tensor_item $loss]
    
    # NLL loss should be positive
    expr {$result_value > 0}
} 1

# Test 2: NLL loss with named parameter syntax
test nll_loss-2.1 {NLL loss with named parameter syntax} {
    # Create test tensors
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -2.0 -1.0 -3.0} -shape {2 3} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]
    
    # Use named parameter syntax
    set loss [torch::nll_loss -input $input -target $target]
    
    set result_value [torch::tensor_item $loss]
    
    expr {$result_value > 0}
} 1

# Test 3: CamelCase alias test
test nll_loss-3.1 {CamelCase alias torch::nllLoss} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -2.0 -1.0 -3.0} -shape {2 3} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]
    
    # Use camelCase alias
    set loss [torch::nllLoss $input $target]
    
    set result_value [torch::tensor_item $loss]
    
    expr {$result_value > 0}
} 1

# Test 4: Error handling - invalid tensor names
test nll_loss-4.1 {Error handling with invalid input tensor} {
    set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]
    
    set result [catch {torch::nll_loss invalid_tensor $target} error]
    
    # Should return error (1)
    expr {$result == 1}
} 1

# Test 5: Error handling - missing required parameters
test nll_loss-5.1 {Error handling with missing parameters} {
    set result [catch {torch::nll_loss} error]
    
    # Should return error for missing arguments
    expr {$result == 1}
} 1

# Test 6: Error handling - invalid named parameter
test nll_loss-6.1 {Error handling with invalid named parameter} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -2.0 -1.0 -3.0} -shape {2 3} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]
    
    set result [catch {torch::nll_loss -input $input -target $target -invalid_param value} error]
    
    expr {$result == 1}
} 1

# Test 7: NLL loss with weight tensor (positional)
test nll_loss-7.1 {NLL loss with weight tensor using positional syntax} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -2.0 -1.0 -3.0} -shape {2 3} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]
    set weight [torch::tensor_create -data {1.0 2.0 0.5} -dtype float32 -device cpu -requiresGrad false]
    
    set loss [torch::nll_loss $input $target $weight]
    
    set result_value [torch::tensor_item $loss]
    
    expr {$result_value > 0}
} 1

# Test 8: NLL loss with weight tensor (named)
test nll_loss-8.1 {NLL loss with weight tensor using named syntax} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -2.0 -1.0 -3.0} -shape {2 3} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]
    set weight [torch::tensor_create -data {1.0 2.0 0.5} -dtype float32 -device cpu -requiresGrad false]
    
    set loss [torch::nll_loss -input $input -target $target -weight $weight]
    
    set result_value [torch::tensor_item $loss]
    
    expr {$result_value > 0}
} 1

# Test 9: NLL loss with 'none' reduction (positional)
test nll_loss-9.1 {NLL loss with none reduction using positional syntax} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -2.0 -1.0 -3.0} -shape {2 3} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]
    
    set loss [torch::nll_loss $input $target none none]
    
    # With 'none' reduction, we should get a tensor with 2 elements (batch size)
    set shape [torch::tensor_shape $loss]
    set shape_list [split $shape " "]
    set first_dim [lindex $shape_list 0]
    
    # Check if first dimension is 2 (batch size)
    expr {$first_dim == 2}
} 1

# Test 10: NLL loss with 'none' reduction (named)
test nll_loss-10.1 {NLL loss with none reduction using named syntax} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -2.0 -1.0 -3.0} -shape {2 3} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]
    
    set loss [torch::nll_loss -input $input -target $target -reduction none]
    
    # With 'none' reduction, we should get a tensor with 2 elements (batch size)
    set shape [torch::tensor_shape $loss]
    set shape_list [split $shape " "]
    set first_dim [lindex $shape_list 0]
    
    expr {$first_dim == 2}
} 1

# Test 11: NLL loss with 'sum' reduction
test nll_loss-11.1 {NLL loss with sum reduction} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -2.0 -1.0 -3.0} -shape {2 3} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]
    
    set loss_mean [torch::nll_loss -input $input -target $target -reduction mean]
    set loss_sum [torch::nll_loss -input $input -target $target -reduction sum]
    
    set mean_value [torch::tensor_item $loss_mean]
    set sum_value [torch::tensor_item $loss_sum]
    
    # Sum should be approximately 2 * mean for batch size 2
    expr {abs($sum_value - (2.0 * $mean_value)) < 0.01}
} 1

# Test 12: Syntax consistency - both syntaxes produce same result
test nll_loss-12.1 {Syntax consistency between positional and named parameters} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -2.0 -1.0 -3.0} -shape {2 3} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]
    
    # Test with positional syntax
    set loss1 [torch::nll_loss $input $target]
    
    # Test with named syntax
    set loss2 [torch::nll_loss -input $input -target $target]
    
    set value1 [torch::tensor_item $loss1]
    set value2 [torch::tensor_item $loss2]
    
    # Results should be identical
    expr {abs($value1 - $value2) < 1e-6}
} 1

# Test 13: Edge case - single sample
test nll_loss-13.1 {NLL loss with single sample} {
    # For single sample, input should still be 2D: [1, num_classes]
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0} -shape {1 3} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {1} -dtype int64 -device cpu -requiresGrad false]
    
    set loss [torch::nll_loss $input $target]
    
    set result_value [torch::tensor_item $loss]
    
    # Should return expected NLL value (2.0 for this input)
    expr {abs($result_value - 2.0) < 1e-5}
} 1

# Test 14: NLL Loss properties - mathematical correctness
test nll_loss-14.1 {NLL Loss mathematical properties} {
    # Create input with perfect predictions (class 0 has highest log probability)
    # Input should be 2D: [1, num_classes]
    set input1 [torch::tensor_create -data {-0.1 -2.0 -3.0} -shape {1 3} -dtype float32 -device cpu -requiresGrad false]
    set target1 [torch::tensor_create -data {0} -dtype int64 -device cpu -requiresGrad false]
    
    # Create input with poor predictions (class 0 has lowest log probability) 
    set input2 [torch::tensor_create -data {-3.0 -0.1 -2.0} -shape {1 3} -dtype float32 -device cpu -requiresGrad false]
    set target2 [torch::tensor_create -data {0} -dtype int64 -device cpu -requiresGrad false]
    
    set loss1 [torch::nll_loss $input1 $target1]
    set loss2 [torch::nll_loss $input2 $target2]
    
    set value1 [torch::tensor_item $loss1]
    set value2 [torch::tensor_item $loss2]
    
    # Better prediction (higher log probability) should have lower loss
    expr {$value1 < $value2}
} 1

# Test 15: Multiple reduction types consistency
test nll_loss-15.1 {Multiple reduction types consistency} {
    set input [torch::tensor_create -data {-1.0 -2.0 -3.0 -2.0 -1.0 -3.0} -shape {2 3} -dtype float32 -device cpu -requiresGrad false]
    set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]
    
    set loss_none [torch::nll_loss -input $input -target $target -reduction none]
    set loss_mean [torch::nll_loss -input $input -target $target -reduction mean]
    set loss_sum [torch::nll_loss -input $input -target $target -reduction sum]
    
    # Check that 'none' reduction has correct shape (batch size = 2)
    set none_shape [torch::tensor_shape $loss_none]
    set none_shape_list [split $none_shape " "]
    set batch_size [lindex $none_shape_list 0]
    
    set mean_value [torch::tensor_item $loss_mean]
    set sum_value [torch::tensor_item $loss_sum]
    
    # Verify relationships: sum should be larger than mean for batch size > 1
    # and none reduction should preserve batch dimension
    expr {$batch_size == 2 && $sum_value > $mean_value && $mean_value > 0 && $sum_value > 0}
} 1

cleanupTests 