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
test cross_entropy_loss-1.1 {Basic positional syntax} {
    # Create logits for 3 classes, 2 samples
    set logits [torch::tensor_create {2.0 1.0 0.1 0.5 3.0 1.5} -dtype float32]
    set logits [torch::tensor_reshape $logits {2 3}]
    
    # Create target class indices
    set target [torch::tensor_create {0 2} int64]
    
    set loss [torch::cross_entropy_loss $logits $target]
    
    set result [torch::tensor_item $loss]
    expr {$result > 0.0}  ; # Loss should be positive
} {1}

# Test 2: Named parameter syntax
test cross_entropy_loss-2.1 {Named parameter syntax} {
    # Create logits for 3 classes, 2 samples
    set logits [torch::tensor_create {2.0 1.0 0.1 0.5 3.0 1.5} -dtype float32]
    set logits [torch::tensor_reshape $logits {2 3}]
    
    # Create target class indices
    set target [torch::tensor_create {0 2} int64]
    
    set loss [torch::cross_entropy_loss -input $logits -target $target]
    
    set result [torch::tensor_item $loss]
    expr {$result > 0.0}  ; # Loss should be positive
} {1}

# Test 3: CamelCase alias
test cross_entropy_loss-3.1 {CamelCase alias} {
    # Create logits for 3 classes, 2 samples
    set logits [torch::tensor_create {2.0 1.0 0.1 0.5 3.0 1.5} -dtype float32]
    set logits [torch::tensor_reshape $logits {2 3}]
    
    # Create target class indices
    set target [torch::tensor_create {0 2} int64]
    
    set loss [torch::crossEntropyLoss -input $logits -target $target]
    
    set result [torch::tensor_item $loss]
    expr {$result > 0.0}  ; # Loss should be positive
} {1}

# Test 4: Positional syntax with reduction
test cross_entropy_loss-4.1 {Positional syntax with reduction} {
    # Create logits for 3 classes, 2 samples
    set logits [torch::tensor_create {2.0 1.0 0.1 0.5 3.0 1.5} -dtype float32]
    set logits [torch::tensor_reshape $logits {2 3}]
    
    # Create target class indices
    set target [torch::tensor_create {0 2} int64]
    
    set loss [torch::cross_entropy_loss $logits $target none sum]
    
    set result [torch::tensor_item $loss]
    expr {$result > 0.0}  ; # Loss should be positive
} {1}

# Test 5: Named syntax with reduction
test cross_entropy_loss-5.1 {Named syntax with reduction} {
    # Create logits for 3 classes, 2 samples
    set logits [torch::tensor_create {2.0 1.0 0.1 0.5 3.0 1.5} -dtype float32]
    set logits [torch::tensor_reshape $logits {2 3}]
    
    # Create target class indices
    set target [torch::tensor_create {0 2} int64]
    
    set loss [torch::cross_entropy_loss -input $logits -target $target -reduction sum]
    
    set result [torch::tensor_item $loss]
    expr {$result > 0.0}  ; # Loss should be positive
} {1}

# Test 6: Named syntax with class weights
test cross_entropy_loss-6.1 {Named syntax with class weights} {
    # Create logits for 3 classes, 2 samples
    set logits [torch::tensor_create {2.0 1.0 0.1 0.5 3.0 1.5} -dtype float32]
    set logits [torch::tensor_reshape $logits {2 3}]
    
    # Create target class indices
    set target [torch::tensor_create {0 2} int64]
    
    # Create class weights (3 classes)
    set weight [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
    
    set loss [torch::cross_entropy_loss -input $logits -target $target -weight $weight]
    
    set result [torch::tensor_item $loss]
    expr {$result > 0.0}  ; # Loss should be positive
} {1}

# Test 7: Test perfect prediction (low loss)
test cross_entropy_loss-7.1 {Test perfect prediction} {
    # Create very confident correct predictions
    set logits [torch::tensor_create {10.0 -5.0 -5.0 -5.0 10.0 -5.0} -dtype float32]
    set logits [torch::tensor_reshape $logits {2 3}]
    
    # Targets match the max logits (class 0 and 1)
    set target [torch::tensor_create {0 1} int64]
    
    set loss [torch::cross_entropy_loss -input $logits -target $target]
    
    set result [torch::tensor_item $loss]
    expr {$result < 0.1}  ; # Loss should be very small for perfect predictions
} {1}

# Test 8: Error handling - missing required parameter
test cross_entropy_loss-8.1 {Error handling - missing required parameter} {
    set logits [torch::tensor_create {2.0 1.0 0.1} -dtype float32]
    catch {torch::cross_entropy_loss -input $logits} result
    string match "*Required parameters*" $result
} {1}

# Test 9: Error handling - invalid parameter name
test cross_entropy_loss-9.1 {Error handling - invalid parameter name} {
    set logits [torch::tensor_create {2.0 1.0 0.1} -dtype float32]
    set target [torch::tensor_create {0} int64]
    catch {torch::cross_entropy_loss -input $logits -target $target -invalid param} result
    string match "*Unknown parameter*" $result
} {1}

# Test 10: Syntax consistency - both syntaxes produce same result
test cross_entropy_loss-10.1 {Syntax consistency} {
    # Create logits for 4 classes, 1 sample
    set logits [torch::tensor_create {1.0 2.0 3.0 0.5} -dtype float32]
    set logits [torch::tensor_reshape $logits {1 4}]
    
    # Create target
    set target [torch::tensor_create {2} int64]
    
    set loss1 [torch::cross_entropy_loss $logits $target]
    set loss2 [torch::cross_entropy_loss -input $logits -target $target]
    
    set result1 [torch::tensor_item $loss1]
    set result2 [torch::tensor_item $loss2]
    
    expr {abs($result1 - $result2) < 1e-6}
} {1}

# Test 11: Test with none reduction
test cross_entropy_loss-11.1 {Test with none reduction} {
    # Create logits for 3 classes, 2 samples
    set logits [torch::tensor_create {2.0 1.0 0.1 0.5 3.0 1.5} -dtype float32]
    set logits [torch::tensor_reshape $logits {2 3}]
    
    # Create target class indices
    set target [torch::tensor_create {0 2} int64]
    
    set loss [torch::cross_entropy_loss -input $logits -target $target -reduction none]
    
    # Should return a tensor with 2 elements (one per sample)
    set shape [torch::tensor_shape $loss]
    list [llength $shape] [lindex $shape 0]
} {1 2}

# Test 12: Single class case
test cross_entropy_loss-12.1 {Single class case} {
    # Create logits for 1 class (binary but using cross-entropy format)
    set logits [torch::tensor_create {2.5 -1.0} -dtype float32]
    set logits [torch::tensor_reshape $logits {2 1}]
    
    # Create targets
    set target [torch::tensor_create {0 0} int64]
    
    set loss [torch::cross_entropy_loss -input $logits -target $target]
    
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0}  ; # Loss should be non-negative
} {1}

cleanupTests 