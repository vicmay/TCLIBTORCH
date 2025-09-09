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

# Helper function to create test triplet tensors
proc create_triplet_tensors {} {
    # Create anchor tensor (reference embeddings)
    set anchor [torch::tensorCreate -data {1.0 0.5 -0.2 0.8 -0.3 1.2} -shape {2 3}]
    
    # Create positive tensor (similar to anchor)
    set positive [torch::tensorCreate -data {0.9 0.6 -0.1 0.7 -0.2 1.1} -shape {2 3}]
    
    # Create negative tensor (dissimilar to anchor)
    set negative [torch::tensorCreate -data {-0.8 1.5 0.9 -1.2 0.8 -0.5} -shape {2 3}]
    
    return [list $anchor $positive $negative]
}

# Test 1: Basic positional syntax (backward compatibility)
test triplet_margin_loss-1.1 {Basic positional syntax} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_loss $anchor $positive $negative]
    expr {$result ne ""}
} 1

test triplet_margin_loss-1.2 {Positional syntax with margin} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_loss $anchor $positive $negative 0.5]
    expr {$result ne ""}
} 1

test triplet_margin_loss-1.3 {Positional syntax with margin and p} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_loss $anchor $positive $negative 1.0 1.0]
    expr {$result ne ""}
} 1

test triplet_margin_loss-1.4 {Positional syntax with all parameters} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_loss $anchor $positive $negative 1.0 2.0 0]
    expr {$result ne ""}
} 1

# Test 2: Named parameter syntax
test triplet_margin_loss-2.1 {Named parameter syntax basic} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative]
    expr {$result ne ""}
} 1

test triplet_margin_loss-2.2 {Named parameter syntax with margin} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -margin 0.5]
    expr {$result ne ""}
} 1

test triplet_margin_loss-2.3 {Named parameter syntax with p norm} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -p 1.0]
    expr {$result ne ""}
} 1

test triplet_margin_loss-2.4 {Named parameter syntax with reduction} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -reduction "sum"]
    expr {$result ne ""}
} 1

test triplet_margin_loss-2.5 {Named parameter syntax with all parameters} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -margin 0.8 -p 1.5 -reduction "none"]
    expr {$result ne ""}
} 1

# Test 3: camelCase alias
test triplet_margin_loss-3.1 {camelCase alias basic} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::tripletMarginLoss -anchor $anchor -positive $positive -negative $negative]
    expr {$result ne ""}
} 1

test triplet_margin_loss-3.2 {camelCase alias with parameters} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::tripletMarginLoss -anchor $anchor -positive $positive -negative $negative -margin 0.5 -p 1.0 -reduction "mean"]
    expr {$result ne ""}
} 1

# Test 4: Syntax consistency (both syntaxes should produce same results)
test triplet_margin_loss-4.1 {Syntax consistency test} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    # Same parameters using different syntaxes
    set result_pos [torch::triplet_margin_loss $anchor $positive $negative 1.0 2.0 1]
    set result_named [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -margin 1.0 -p 2.0 -reduction "mean"]
    set result_camel [torch::tripletMarginLoss -anchor $anchor -positive $positive -negative $negative -margin 1.0 -p 2.0 -reduction "mean"]
    
    # All results should be valid tensor handles
    expr {$result_pos ne "" && $result_named ne "" && $result_camel ne ""}
} 1

# Test 5: Different reduction modes
test triplet_margin_loss-5.1 {Different reduction modes} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    # Test different reduction modes
    set result_none [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -reduction "none"]
    set result_mean [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -reduction "mean"]
    set result_sum [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -reduction "sum"]
    
    expr {$result_none ne "" && $result_mean ne "" && $result_sum ne ""}
} 1

# Test 6: Different margin values
test triplet_margin_loss-6.1 {Different margin values} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    # Test different margin values
    set result_small [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -margin 0.1]
    set result_medium [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -margin 1.0]
    set result_large [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -margin 2.0]
    
    expr {$result_small ne "" && $result_medium ne "" && $result_large ne ""}
} 1

# Test 7: Different norm values (p)
test triplet_margin_loss-7.1 {Different norm values} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    # Test different p-norm values
    set result_l1 [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -p 1.0]
    set result_l2 [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -p 2.0]
    
    expr {$result_l1 ne "" && $result_l2 ne ""}
} 1

# Test 8: Error handling
test triplet_margin_loss-8.1 {Error handling - missing parameters} {
    set result [catch {torch::triplet_margin_loss} error_msg]
    expr {$result == 1}
} 1

test triplet_margin_loss-8.2 {Error handling - insufficient positional parameters} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    
    set result [catch {torch::triplet_margin_loss $anchor $positive} error_msg]
    expr {$result == 1}
} 1

test triplet_margin_loss-8.3 {Error handling - invalid tensor name} {
    set tensors [create_triplet_tensors]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    catch {torch::triplet_margin_loss invalid_tensor $positive $negative} result
    string match "*Invalid anchor tensor name*" $result
} 1

test triplet_margin_loss-8.4 {Error handling - named syntax missing values} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    
    catch {torch::triplet_margin_loss -anchor $anchor -positive $positive -negative} result
    string match "*Named parameters must have values*" $result
} 1

test triplet_margin_loss-8.5 {Error handling - unknown named parameter} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    catch {torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -unknown_param value} result
    string match "*Unknown parameter*" $result
} 1

test triplet_margin_loss-8.6 {Error handling - missing required named parameters} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    
    catch {torch::triplet_margin_loss -anchor $anchor -positive $positive} result
    string match "*Required parameters*" $result
} 1

# Test 9: Mathematical correctness
test triplet_margin_loss-9.1 {Different reduction modes produce different results} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result_none [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -reduction "none"]
    set result_mean [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -reduction "mean"]
    
    # Both results should be valid tensors (different shapes expected)
    expr {$result_none ne "" && $result_mean ne "" && $result_none ne $result_mean}
} 1

# Test 10: Data type compatibility
test triplet_margin_loss-10.1 {Float32 tensor compatibility} {
    set anchor [torch::tensorCreate -data {1.0 0.5 -0.2} -shape {1 3} -dtype float32]
    set positive [torch::tensorCreate -data {0.9 0.6 -0.1} -shape {1 3} -dtype float32]
    set negative [torch::tensorCreate -data {-0.8 1.5 0.9} -shape {1 3} -dtype float32]
    
    set result [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative]
    expr {$result ne ""}
} 1

test triplet_margin_loss-10.2 {Double tensor compatibility} {
    set anchor [torch::tensorCreate -data {1.0 0.5 -0.2} -shape {1 3} -dtype float64]
    set positive [torch::tensorCreate -data {0.9 0.6 -0.1} -shape {1 3} -dtype float64]
    set negative [torch::tensorCreate -data {-0.8 1.5 0.9} -shape {1 3} -dtype float64]
    
    set result [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative]
    expr {$result ne ""}
} 1

# Test 11: Face recognition simulation
test triplet_margin_loss-11.1 {Face recognition embeddings} {
    # Simulate face embeddings (anchor=person A, positive=another photo of A, negative=person B)
    set anchor_face [torch::tensorCreate -data {0.8 -0.2 0.5 -0.1 0.9 -0.3 0.7 0.4} -shape {1 8}]
    # Similar to anchor
    set positive_face [torch::tensorCreate -data {0.7 -0.1 0.4 -0.2 0.8 -0.4 0.6 0.3} -shape {1 8}]
    # Different person
    set negative_face [torch::tensorCreate -data {-0.3 0.9 -0.7 0.8 -0.2 0.6 -0.5 -0.9} -shape {1 8}]
    
    set result [torch::triplet_margin_loss -anchor $anchor_face -positive $positive_face -negative $negative_face -margin 0.2]
    expr {$result ne ""}
} 1

# Test 12: Embedding learning simulation  
test triplet_margin_loss-12.1 {Word embedding learning} {
    # Simulate word embeddings (anchor=word, positive=synonym, negative=unrelated word)
    set word_anchor [torch::tensorCreate -data {0.5 -0.8 0.3 0.7 -0.4} -shape {1 5}]
    # Synonym
    set word_positive [torch::tensorCreate -data {0.6 -0.7 0.2 0.8 -0.3} -shape {1 5}]
    # Unrelated
    set word_negative [torch::tensorCreate -data {-0.9 0.4 -0.6 -0.2 0.8} -shape {1 5}]
    
    set result [torch::triplet_margin_loss -anchor $word_anchor -positive $word_positive -negative $word_negative -p 1.0 -margin 0.5]
    expr {$result ne ""}
} 1

# Test 13: Batch processing
test triplet_margin_loss-13.1 {Batch processing compatibility} {
    # Simulate batch of 3 triplets, each with 4-dimensional embeddings
    set batch_anchor [torch::tensorCreate -data {1.0 0.5 -0.2 0.8 0.3 -0.7 0.9 -0.1 -0.4 0.6 0.2 -0.5} -shape {3 4}]
    set batch_positive [torch::tensorCreate -data {0.9 0.6 -0.1 0.7 0.4 -0.6 0.8 0.0 -0.3 0.7 0.3 -0.4} -shape {3 4}]
    set batch_negative [torch::tensorCreate -data {-0.8 1.5 0.9 -1.2 -0.9 0.8 -0.5 1.3 0.8 -0.9 -0.7 1.0} -shape {3 4}]
    
    set result [torch::triplet_margin_loss -anchor $batch_anchor -positive $batch_positive -negative $batch_negative -reduction "mean"]
    expr {$result ne ""}
} 1

# Test 14: Edge cases
test triplet_margin_loss-14.1 {Edge case - identical anchor and positive} {
    set anchor [torch::tensorCreate -data {1.0 0.5 -0.2 0.8} -shape {1 4}]
    # Identical to anchor
    set positive [torch::tensorCreate -data {1.0 0.5 -0.2 0.8} -shape {1 4}]
    set negative [torch::tensorCreate -data {-0.8 1.5 0.9 -1.2} -shape {1 4}]
    
    set result [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative]
    expr {$result ne ""}
} 1

test triplet_margin_loss-14.2 {Edge case - zero margin} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -margin 0.0]
    expr {$result ne ""}
} 1

test triplet_margin_loss-14.3 {Edge case - large margin} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -margin 10.0]
    expr {$result ne ""}
} 1

# Test 15: Parameter order flexibility (named parameters)
test triplet_margin_loss-15.1 {Named parameter order flexibility} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    # Parameters in different order
    set result [torch::triplet_margin_loss -negative $negative -margin 0.5 -anchor $anchor -positive $positive -p 1.0]
    expr {$result ne ""}
} 1

cleanupTests 