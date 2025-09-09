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

# Test 1: Basic hinge embedding loss with positional syntax
test hinge_embedding_loss-1.1 {Basic hinge embedding loss with positional syntax} {
    # Create input and target tensors
    set input [torch::tensor_create {1.0 -1.5 2.0 -0.5} float32]
    set target [torch::tensor_create {1 -1 1 -1} float32]
    
    # Compute hinge embedding loss using positional syntax
    set loss [torch::hinge_embedding_loss $input $target]
    
    set result_value [torch::tensor_item $loss]
    
    # Hinge embedding loss should be non-negative
    expr {$result_value >= 0}
} 1

# Test 2: Hinge embedding loss with named parameter syntax
test hinge_embedding_loss-2.1 {Hinge embedding loss with named parameter syntax} {
    set input [torch::tensor_create {1.0 -1.5 2.0 -0.5} float32]
    set target [torch::tensor_create {1 -1 1 -1} float32]
    
    # Use named parameter syntax
    set loss [torch::hinge_embedding_loss -input $input -target $target]
    
    set result_value [torch::tensor_item $loss]
    
    expr {$result_value >= 0}
} 1

# Test 3: CamelCase alias test
test hinge_embedding_loss-3.1 {CamelCase alias torch::hingeEmbeddingLoss} {
    set input [torch::tensor_create {1.0 -1.5 2.0 -0.5} float32]
    set target [torch::tensor_create {1 -1 1 -1} float32]
    
    # Use camelCase alias
    set loss [torch::hingeEmbeddingLoss $input $target]
    
    set result_value [torch::tensor_item $loss]
    
    expr {$result_value >= 0}
} 1

# Test 4: Error handling - invalid tensor names
test hinge_embedding_loss-4.1 {Error handling with invalid input tensor} {
    set target [torch::tensor_create {1 -1} float32]
    
    set result [catch {torch::hinge_embedding_loss invalid_tensor $target} error]
    
    # Should return error (1)
    expr {$result == 1}
} 1

# Test 5: Error handling - missing required parameters
test hinge_embedding_loss-5.1 {Error handling with missing parameters} {
    set result [catch {torch::hinge_embedding_loss} error]
    
    # Should return error for missing arguments
    expr {$result == 1}
} 1

# Test 6: Error handling - invalid named parameter
test hinge_embedding_loss-6.1 {Error handling with invalid named parameter} {
    set input [torch::tensor_create {1.0 -1.5} float32]
    set target [torch::tensor_create {1 -1} float32]
    
    set result [catch {torch::hinge_embedding_loss -input $input -target $target -invalid_param value} error]
    
    expr {$result == 1}
} 1

# Test 7: Hinge embedding loss with custom margin (positional)
test hinge_embedding_loss-7.1 {Hinge embedding loss with custom margin using positional syntax} {
    set input [torch::tensor_create {0.5 -0.8 1.2 -0.3} float32]
    set target [torch::tensor_create {1 -1 1 -1} float32]
    
    # Use margin of 2.0
    set loss [torch::hinge_embedding_loss $input $target 2.0]
    
    set result_value [torch::tensor_item $loss]
    
    expr {$result_value >= 0}
} 1

# Test 8: Hinge embedding loss with custom margin (named)
test hinge_embedding_loss-8.1 {Hinge embedding loss with custom margin using named syntax} {
    set input [torch::tensor_create {0.5 -0.8 1.2 -0.3} float32]
    set target [torch::tensor_create {1 -1 1 -1} float32]
    
    set loss [torch::hinge_embedding_loss -input $input -target $target -margin 2.0]
    
    set result_value [torch::tensor_item $loss]
    
    expr {$result_value >= 0}
} 1

# Test 9: Hinge embedding loss with 'none' reduction (positional)
test hinge_embedding_loss-9.1 {Hinge embedding loss with none reduction using positional syntax} {
    set input [torch::tensor_create {1.0 -1.5 2.0 -0.5} float32]
    set target [torch::tensor_create {1 -1 1 -1} float32]
    
    set loss [torch::hinge_embedding_loss $input $target 1.0 none]
    
    # With 'none' reduction, we should get a tensor with 4 elements (input size)
    set shape [torch::tensor_shape $loss]
    set shape_list [split $shape " "]
    set first_dim [lindex $shape_list 0]
    
    # Check if first dimension is 4 (number of elements)
    expr {$first_dim == 4}
} 1

# Test 10: Hinge embedding loss with 'none' reduction (named)
test hinge_embedding_loss-10.1 {Hinge embedding loss with none reduction using named syntax} {
    set input [torch::tensor_create {1.0 -1.5 2.0 -0.5} float32]
    set target [torch::tensor_create {1 -1 1 -1} float32]
    
    set loss [torch::hinge_embedding_loss -input $input -target $target -reduction none]
    
    # With 'none' reduction, we should get a tensor with 4 elements
    set shape [torch::tensor_shape $loss]
    set shape_list [split $shape " "]
    set first_dim [lindex $shape_list 0]
    
    expr {$first_dim == 4}
} 1

# Test 11: Hinge embedding loss with 'sum' reduction
test hinge_embedding_loss-11.1 {Hinge embedding loss with sum reduction} {
    set input [torch::tensor_create {1.0 -1.5 2.0 -0.5} float32]
    set target [torch::tensor_create {1 -1 1 -1} float32]
    
    set loss_mean [torch::hinge_embedding_loss -input $input -target $target -reduction mean]
    set loss_sum [torch::hinge_embedding_loss -input $input -target $target -reduction sum]
    
    set mean_value [torch::tensor_item $loss_mean]
    set sum_value [torch::tensor_item $loss_sum]
    
    # Sum should be larger than mean for multiple elements
    expr {$sum_value >= $mean_value}
} 1

# Test 12: Syntax consistency - both syntaxes produce same result
test hinge_embedding_loss-12.1 {Syntax consistency between positional and named parameters} {
    set input [torch::tensor_create {0.8 -1.2 1.5 -0.7} float32]
    set target [torch::tensor_create {1 -1 1 -1} float32]
    
    # Test with positional syntax
    set loss1 [torch::hinge_embedding_loss $input $target]
    
    # Test with named syntax
    set loss2 [torch::hinge_embedding_loss -input $input -target $target]
    
    set value1 [torch::tensor_item $loss1]
    set value2 [torch::tensor_item $loss2]
    
    # Results should be identical
    expr {abs($value1 - $value2) < 1e-6}
} 1

# Test 13: Hinge embedding loss mathematical properties - target = 1
test hinge_embedding_loss-13.1 {Hinge embedding loss with target = 1} {
    # When target is 1, loss = input (regardless of sign)
    set input1 [torch::tensor_create -data {2.0} -dtype float32]
    set target1 [torch::tensor_create -data {1} -dtype float32]
    
    set input2 [torch::tensor_create -data {-0.5} -dtype float32]
    set target2 [torch::tensor_create -data {1} -dtype float32]
    
    set loss1 [torch::hinge_embedding_loss $input1 $target1]
    set loss2 [torch::hinge_embedding_loss $input2 $target2]
    
    set value1 [torch::tensor_item $loss1]
    set value2 [torch::tensor_item $loss2]
    
    # For target=1, loss should equal input value
    expr {abs($value1 - 2.0) < 1e-5 && abs($value2 - (-0.5)) < 1e-5}
} 1

# Test 14: Hinge embedding loss mathematical properties - target = -1
test hinge_embedding_loss-14.1 {Hinge embedding loss with target = -1} {
    # When target is -1, loss = max(0, margin - input)
    set input1 [torch::tensor_create -data {-2.0} -dtype float32]
    set target1 [torch::tensor_create -data {-1} -dtype float32]
    
    set input2 [torch::tensor_create -data {0.5} -dtype float32]
    set target2 [torch::tensor_create -data {-1} -dtype float32]
    
    set loss1 [torch::hinge_embedding_loss $input1 $target1 1.0]  ; # max(0, 1-(-2)) = max(0,3) = 3
    set loss2 [torch::hinge_embedding_loss $input2 $target2 1.0]  ; # max(0, 1-0.5) = max(0,0.5) = 0.5
    
    set value1 [torch::tensor_item $loss1]
    set value2 [torch::tensor_item $loss2]
    
    # For target=-1, loss = max(0, margin - input)
    expr {abs($value1 - 3.0) < 1e-5 && abs($value2 - 0.5) < 1e-5}
} 1

# Test 15: Different margin values effect
test hinge_embedding_loss-15.1 {Effect of different margin values} {
    set input [torch::tensor_create -data {0.5} -dtype float32]
    set target [torch::tensor_create -data {-1} -dtype float32]
    
    # Larger margin should result in larger loss for dissimilar pairs
    set loss_small_margin [torch::hingeEmbeddingLoss -input $input -target $target -margin 0.5]
    set loss_large_margin [torch::hingeEmbeddingLoss -input $input -target $target -margin 2.0]
    
    set small_value [torch::tensor_item $loss_small_margin]
    set large_value [torch::tensor_item $loss_large_margin]
    
    # Larger margin should give larger loss for dissimilar pairs
    expr {$large_value >= $small_value}
} 1

# Test 16: Backward compatibility with integer reduction
test hinge_embedding_loss-16.1 {Backward compatibility with integer reduction} {
    set input [torch::tensor_create {1.0 -1.5} float32]
    set target [torch::tensor_create {1 -1} float32]
    
    # Test integer reduction (0=none, 1=mean, 2=sum)
    set loss_int [torch::hinge_embedding_loss $input $target 1.0 2]  ; # 2 = sum
    set loss_str [torch::hinge_embedding_loss -input $input -target $target -reduction sum]
    
    set int_value [torch::tensor_item $loss_int]
    set str_value [torch::tensor_item $loss_str]
    
    # Results should be identical
    expr {abs($int_value - $str_value) < 1e-6}
} 1

cleanupTests 