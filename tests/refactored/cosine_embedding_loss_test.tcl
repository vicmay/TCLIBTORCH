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
test cosine_embedding_loss-1.1 {Basic positional syntax} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
    set input1 [torch::tensor_reshape $input1 {1 3}]
    set input2 [torch::tensor_create {2.0 3.0 4.0} -dtype float32]
    set input2 [torch::tensor_reshape $input2 {1 3}]
    set target [torch::tensor_create {1.0} -dtype float32]
    set loss [torch::cosine_embedding_loss $input1 $input2 $target]
    
    # Should return a scalar loss value
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0}  ; # Loss should be non-negative
} {1}

# Test 2: Named parameter syntax
test cosine_embedding_loss-2.1 {Named parameter syntax} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
    set input1 [torch::tensor_reshape $input1 {1 3}]
    set input2 [torch::tensor_create {2.0 3.0 4.0} -dtype float32]
    set input2 [torch::tensor_reshape $input2 {1 3}]
    set target [torch::tensor_create {1.0} -dtype float32]
    set loss [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target]
    
    # Should return a scalar loss value
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0}  ; # Loss should be non-negative
} {1}

# Test 3: CamelCase alias
test cosine_embedding_loss-3.1 {CamelCase alias} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
    set input1 [torch::tensor_reshape $input1 {1 3}]
    set input2 [torch::tensor_create {2.0 3.0 4.0} -dtype float32]
    set input2 [torch::tensor_reshape $input2 {1 3}]
    set target [torch::tensor_create {1.0} -dtype float32]
    set loss [torch::cosineEmbeddingLoss -input1 $input1 -input2 $input2 -target $target]
    
    # Should return a scalar loss value
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0}  ; # Loss should be non-negative
} {1}

# Test 4: Positional syntax with margin
test cosine_embedding_loss-4.1 {Positional syntax with margin} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
    set input1 [torch::tensor_reshape $input1 {1 3}]
    set input2 [torch::tensor_create {2.0 3.0 4.0} -dtype float32]
    set input2 [torch::tensor_reshape $input2 {1 3}]
    set target [torch::tensor_create {1.0} -dtype float32]
    set loss [torch::cosine_embedding_loss $input1 $input2 $target 0.5]
    
    # Should return a scalar loss value
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0}  ; # Loss should be non-negative
} {1}

# Test 5: Named syntax with margin and reduction
test cosine_embedding_loss-5.1 {Named syntax with margin and reduction} {
    set input1 [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32]
    set input1 [torch::tensor_reshape $input1 {2 3}]
    set input2 [torch::tensor_create {2.0 3.0 4.0 5.0 6.0 7.0} -dtype float32]
    set input2 [torch::tensor_reshape $input2 {2 3}]
    set target [torch::tensor_create {1.0 -1.0} -dtype float32]
    set loss [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target -margin 0.3 -reduction sum]
    
    # Should return a scalar loss value
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0}  ; # Loss should be non-negative
} {1}

# Test 6: Test with negative target values
test cosine_embedding_loss-6.1 {Test with negative target values} {
    set input1 [torch::tensor_create -data {1.0 0.0 0.0} -dtype float32]
    set input1 [torch::tensor_reshape $input1 {1 3}]
    set input2 [torch::tensor_create -data {0.0 1.0 0.0} -dtype float32]
    set input2 [torch::tensor_reshape $input2 {1 3}]
    set target [torch::tensor_create -data {-1.0} -dtype float32]  ; # Dissimilar
    set loss [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target -margin 0.5]
    
    # Should return a scalar loss value
    set result [torch::tensor_item $loss]
    expr {$result >= 0.0}  ; # Loss should be non-negative
} {1}

# Test 7: Error handling - missing required parameter
test cosine_embedding_loss-7.1 {Error handling - missing required parameter} {
    set input1 [torch::tensor_create {1.0 2.0} -dtype float32]
    set input1 [torch::tensor_reshape $input1 {1 2}]
    set input2 [torch::tensor_create {2.0 3.0} -dtype float32]
    set input2 [torch::tensor_reshape $input2 {1 2}]
    catch {torch::cosine_embedding_loss -input1 $input1 -input2 $input2} result
    string match "*Required parameters*" $result
} {1}

# Test 8: Error handling - invalid parameter name
test cosine_embedding_loss-8.1 {Error handling - invalid parameter name} {
    set input1 [torch::tensor_create {1.0 2.0} -dtype float32]
    set input1 [torch::tensor_reshape $input1 {1 2}]
    set input2 [torch::tensor_create {2.0 3.0} -dtype float32]
    set input2 [torch::tensor_reshape $input2 {1 2}]
    set target [torch::tensor_create {1.0} -dtype float32]
    catch {torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target -invalid param} result
    string match "*Unknown parameter*" $result
} {1}

# Test 9: Syntax consistency - both syntaxes produce same result
test cosine_embedding_loss-9.1 {Syntax consistency} {
    set input1 [torch::tensor_create {3.0 4.0 5.0} -dtype float32]
    set input1 [torch::tensor_reshape $input1 {1 3}]
    set input2 [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
    set input2 [torch::tensor_reshape $input2 {1 3}]
    set target [torch::tensor_create {1.0} -dtype float32]
    
    set loss1 [torch::cosine_embedding_loss $input1 $input2 $target 0.2]
    set loss2 [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target -margin 0.2]
    
    set result1 [torch::tensor_item $loss1]
    set result2 [torch::tensor_item $loss2]
    
    expr {abs($result1 - $result2) < 1e-6}
} {1}

# Test 10: Test with none reduction
test cosine_embedding_loss-10.1 {Test with none reduction} {
    set input1 [torch::tensor_create {1.0 0.0 0.0 0.0 1.0 0.0} -dtype float32]
    set input1 [torch::tensor_reshape $input1 {2 3}]
    set input2 [torch::tensor_create {0.0 1.0 0.0 1.0 0.0 0.0} -dtype float32]
    set input2 [torch::tensor_reshape $input2 {2 3}]
    set target [torch::tensor_create {1.0 -1.0} -dtype float32]
    set loss [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target -reduction none]
    
    # Should return a tensor with 2 elements (one per sample)
    set shape [torch::tensor_shape $loss]
    list [llength $shape] [lindex $shape 0]
} {1 2}

# Test 11: Test with zero margin (default)
test cosine_embedding_loss-11.1 {Test with zero margin default} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
    set input1 [torch::tensor_reshape $input1 {1 3}]
    set input2 [torch::tensor_create {1.0 2.0 3.0} -dtype float32]  ; # Same vectors
    set input2 [torch::tensor_reshape $input2 {1 3}]
    set target [torch::tensor_create {1.0} -dtype float32]  ; # Similar
    set loss [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target]
    
    # Loss should be very small for identical vectors with positive target
    set result [torch::tensor_item $loss]
    expr {$result < 0.1}  ; # Should be close to 0
} {1}

cleanupTests 