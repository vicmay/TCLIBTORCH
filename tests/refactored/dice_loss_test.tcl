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

# Test 1: Basic positional syntax
test dice_loss-1.1 {Basic positional syntax} {
    # Create predicted segmentation map (logits before sigmoid)
    set pred [torch::tensor_create {2.0 -1.0 1.5 -0.5} -dtype float32]
    set pred [torch::tensor_reshape $pred {2 2}]
    
    # Create ground truth binary mask
    set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]
    set target [torch::tensor_reshape $target {2 2}]
    
    set loss [torch::dice_loss $pred $target]
    
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0 && $result <= 1.0}  ; # Dice loss should be between 0 and 1
} {1}

# Test 2: Named parameter syntax
test dice_loss-2.1 {Named parameter syntax} {
    # Create predicted segmentation map
    set pred [torch::tensor_create {2.0 -1.0 1.5 -0.5} -dtype float32]
    set pred [torch::tensor_reshape $pred {2 2}]
    
    # Create ground truth binary mask
    set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]
    set target [torch::tensor_reshape $target {2 2}]
    
    set loss [torch::dice_loss -input $pred -target $target]
    
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0 && $result <= 1.0}  ; # Dice loss should be between 0 and 1
} {1}

# Test 3: CamelCase alias
test dice_loss-3.1 {CamelCase alias} {
    # Create predicted segmentation map
    set pred [torch::tensor_create {2.0 -1.0 1.5 -0.5} -dtype float32]
    set pred [torch::tensor_reshape $pred {2 2}]
    
    # Create ground truth binary mask
    set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]
    set target [torch::tensor_reshape $target {2 2}]
    
    set loss [torch::diceLoss -input $pred -target $target]
    
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0 && $result <= 1.0}  ; # Dice loss should be between 0 and 1
} {1}

# Test 4: Positional syntax with custom smooth parameter
test dice_loss-4.1 {Positional syntax with custom smooth parameter} {
    # Create predicted segmentation map
    set pred [torch::tensor_create {2.0 -1.0 1.5 -0.5} -dtype float32]
    set pred [torch::tensor_reshape $pred {2 2}]
    
    # Create ground truth binary mask
    set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]
    set target [torch::tensor_reshape $target {2 2}]
    
    set loss [torch::dice_loss $pred $target 2.0]
    
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0 && $result <= 1.0}  ; # Dice loss should be between 0 and 1
} {1}

# Test 5: Named syntax with custom smooth parameter
test dice_loss-5.1 {Named syntax with custom smooth parameter} {
    # Create predicted segmentation map
    set pred [torch::tensor_create {2.0 -1.0 1.5 -0.5} -dtype float32]
    set pred [torch::tensor_reshape $pred {2 2}]
    
    # Create ground truth binary mask
    set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]
    set target [torch::tensor_reshape $target {2 2}]
    
    set loss [torch::dice_loss -input $pred -target $target -smooth 2.0]
    
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0 && $result <= 1.0}  ; # Dice loss should be between 0 and 1
} {1}

# Test 6: Named syntax with reduction parameter
test dice_loss-6.1 {Named syntax with reduction parameter} {
    # Create predicted segmentation map
    set pred [torch::tensor_create {2.0 -1.0 1.5 -0.5} -dtype float32]
    set pred [torch::tensor_reshape $pred {2 2}]
    
    # Create ground truth binary mask
    set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]
    set target [torch::tensor_reshape $target {2 2}]
    
    set loss [torch::dice_loss -input $pred -target $target -reduction sum]
    
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0}  ; # Loss should be non-negative
} {1}

# Test 7: Perfect segmentation (should give low loss)
test dice_loss-7.1 {Perfect segmentation} {
    # Create perfect predictions (very confident correct segmentation)
    set pred [torch::tensor_create {10.0 -10.0 10.0 -10.0} -dtype float32]
    set pred [torch::tensor_reshape $pred {2 2}]
    
    # Create matching ground truth
    set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]
    set target [torch::tensor_reshape $target {2 2}]
    
    set loss [torch::dice_loss -input $pred -target $target]
    
    set result [torch::tensor_item $loss]
    expr {$result < 0.1}  ; # Perfect segmentation should have very low loss
} {1}

# Test 8: All background prediction
test dice_loss-8.1 {All background prediction} {
    # Create all background predictions
    set pred [torch::tensor_create -data {-5.0 -5.0 -5.0 -5.0} -dtype float32]
    set pred [torch::tensor_reshape $pred {2 2}]
    
    # Create all background ground truth
    set target [torch::tensor_create {0.0 0.0 0.0 0.0} -dtype float32]
    set target [torch::tensor_reshape $target {2 2}]
    
    set loss [torch::dice_loss -input $pred -target $target]
    
    set result [torch::tensor_item $loss]
    expr {$result < 0.1}  ; # Matching background should have low loss
} {1}

# Test 9: Error handling - missing required parameter
test dice_loss-9.1 {Error handling - missing required parameter} {
    set pred [torch::tensor_create {2.0 -1.0} -dtype float32]
    catch {torch::dice_loss -input $pred} result
    string match "*Required parameters*" $result
} {1}

# Test 10: Error handling - invalid parameter name
test dice_loss-10.1 {Error handling - invalid parameter name} {
    set pred [torch::tensor_create {2.0 -1.0} -dtype float32]
    set target [torch::tensor_create {1.0 0.0} -dtype float32]
    catch {torch::dice_loss -input $pred -target $target -invalid param} result
    string match "*Unknown parameter*" $result
} {1}

# Test 11: Syntax consistency - both syntaxes produce same result
test dice_loss-11.1 {Syntax consistency} {
    # Create test tensors
    set pred [torch::tensor_create {1.0 -0.5 2.0 0.0} -dtype float32]
    set pred [torch::tensor_reshape $pred {2 2}]
    set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]
    set target [torch::tensor_reshape $target {2 2}]
    
    set loss1 [torch::dice_loss $pred $target]
    set loss2 [torch::dice_loss -input $pred -target $target]
    
    set result1 [torch::tensor_item $loss1]
    set result2 [torch::tensor_item $loss2]
    
    expr {abs($result1 - $result2) < 1e-6}
} {1}

# Test 12: Reduction none
test dice_loss-12.1 {Reduction none} {
    # Create predicted segmentation map
    set pred [torch::tensor_create {2.0 -1.0 1.5 -0.5} -dtype float32]
    set pred [torch::tensor_reshape $pred {2 2}]
    
    # Create ground truth binary mask
    set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]
    set target [torch::tensor_reshape $target {2 2}]
    
    set loss [torch::dice_loss -input $pred -target $target -reduction none]
    
    # Should return a tensor (not scalar since it's unsqueezed)
    set shape [torch::tensor_shape $loss]
    expr {[llength $shape] >= 1}
} {1}

# Test 13: Large batch test
test dice_loss-13.1 {Large batch test} {
    # Create batch of segmentation predictions (8x8 image, batch size 1)
    set pred [torch::randn {1 1 8 8}]
    
    # Create binary mask using ones and zeros
    set target [torch::ones {1 1 8 8}]
    set zero_mask [torch::zeros {1 1 4 4}]
    set target [torch::tensor_create {1.0 0.0 1.0 0.0 0.0 1.0 0.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 1.0 1.0 1.0 0.0 1.0 0.0 0.0 1.0 0.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 1.0 1.0 1.0 0.0 1.0 0.0 0.0 1.0 0.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 1.0 1.0 1.0 0.0 1.0 0.0 0.0 1.0 0.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 1.0 1.0} float32]
    set target [torch::tensor_reshape $target {1 1 8 8}]
    
    set loss [torch::diceLoss -input $pred -target $target -smooth 1.0]
    
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0 && $result <= 1.0}  ; # Should be valid dice loss
} {1}

# Test 14: Different smooth values comparison
test dice_loss-14.1 {Different smooth values comparison} {
    # Create predicted segmentation map
    set pred [torch::tensor_create {0.5 -0.5 0.5 -0.5} -dtype float32]
    set pred [torch::tensor_reshape $pred {2 2}]
    
    # Create ground truth binary mask
    set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]
    set target [torch::tensor_reshape $target {2 2}]
    
    set loss_smooth1 [torch::dice_loss -input $pred -target $target -smooth 1.0]
    set loss_smooth10 [torch::dice_loss -input $pred -target $target -smooth 10.0]
    
    set result1 [torch::tensor_item $loss_smooth1]
    set result2 [torch::tensor_item $loss_smooth10]
    
    # Higher smooth should generally result in different loss
    expr {$result1 != $result2}
} {1}

cleanupTests 