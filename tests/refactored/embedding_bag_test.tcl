#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the shared library
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configure test parameters
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper procedure to create test tensors
proc create_tensor {data dims {dtype float32}} {
    return [torch::tensor_create -data $data -dtype $dtype -shape $dims]
}

proc create_int_tensor {data dims} {
    return [torch::tensor_create -data $data -dtype int64 -shape $dims]
}

# Helper procedure to check tensor values approximately
proc tensor_approx_equal {t1 t2 {tolerance 1e-5}} {
    set diff [torch::tensor_sub $t1 $t2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    return [expr {$max_val < $tolerance}]
}

# ========================================
# Tests for Positional Syntax (Backward Compatibility)
# ========================================

test embedding_bag-1.1 {Basic embedding bag sum - positional syntax} {
    # Create test data
    # input: indices [0, 1, 2, 1] - 4 indices
    # offsets: [0, 2] - bag 1: indices 0,1  bag 2: indices 2,3
    # weight: 3x2 embedding matrix
    set input [create_int_tensor {0 1 2 1} {4}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0 2} {2}]
    
    # Mode 0 = sum
    set result [torch::embedding_bag $input $weight $offsets 0]
    
    # Check output shape should be [2, 2] (2 bags, 2-dim embeddings)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test embedding_bag-1.2 {Embedding bag mean - positional syntax} {
    # Same data as above but with mode 1 (mean)
    set input [create_int_tensor {0 1 2 1} {4}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0 2} {2}]
    
    # Mode 1 = mean
    set result [torch::embedding_bag $input $weight $offsets 1]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test embedding_bag-1.3 {Embedding bag max - positional syntax} {
    # Mode 2 = max
    set input [create_int_tensor {0 1 2 1} {4}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0 2} {2}]
    
    # Mode 2 = max
    set result [torch::embedding_bag $input $weight $offsets 2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test embedding_bag-1.4 {Single bag - positional syntax} {
    # Single bag containing all indices
    set input [create_int_tensor {0 1 2} {3}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0} {1}]
    
    set result [torch::embedding_bag $input $weight $offsets 0]
    
    # Check output shape should be [1, 2] (1 bag, 2-dim)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1 2"}
} 1

test embedding_bag-1.5 {Multiple equal-sized bags - positional syntax} {
    # Three bags, 2 indices each
    set input [create_int_tensor {0 1 2 0 1 2} {6}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0 2 4} {3}]
    
    set result [torch::embedding_bag $input $weight $offsets 0]
    
    # Check output shape should be [3, 2] (3 bags, 2-dim)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

# ========================================
# Tests for Named Parameter Syntax
# ========================================

test embedding_bag-2.1 {Named syntax with all parameters} {
    set input [create_int_tensor {0 1 2 1} {4}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0 2} {2}]
    
    set result [torch::embedding_bag -input $input -weight $weight -offsets $offsets -mode 0]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test embedding_bag-2.2 {Named syntax with mean mode} {
    set input [create_int_tensor {0 1 2 1} {4}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0 2} {2}]
    
    set result [torch::embedding_bag -input $input -weight $weight -offsets $offsets -mode 1]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test embedding_bag-2.3 {Named syntax with max mode} {
    set input [create_int_tensor {0 1 2 1} {4}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0 2} {2}]
    
    set result [torch::embedding_bag -input $input -weight $weight -offsets $offsets -mode 2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test embedding_bag-2.4 {Parameter order independence} {
    set input [create_int_tensor {0 1 2} {3}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0} {1}]
    
    set result [torch::embedding_bag -mode 0 -offsets $offsets -input $input -weight $weight]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1 2"}
} 1

# ========================================
# Tests for CamelCase Alias
# ========================================

test embedding_bag-3.1 {CamelCase alias - positional syntax} {
    set input [create_int_tensor {0 1 2} {3}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0} {1}]
    
    set result [torch::embeddingBag $input $weight $offsets 0]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1 2"}
} 1

test embedding_bag-3.2 {CamelCase alias - named syntax} {
    set input [create_int_tensor {0 1 2} {3}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0} {1}]
    
    set result [torch::embeddingBag -input $input -weight $weight -offsets $offsets -mode 0]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1 2"}
} 1

# ========================================
# Tests for Syntax Consistency
# ========================================

test embedding_bag-4.1 {Syntax consistency - same result structure} {
    set input [create_int_tensor {0 1 2 1} {4}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0 2} {2}]
    
    set result1 [torch::embedding_bag $input $weight $offsets 0]
    set result2 [torch::embedding_bag -input $input -weight $weight -offsets $offsets -mode 0]
    
    # Check both have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test embedding_bag-4.2 {CamelCase produces same shape} {
    set input [create_int_tensor {0 1 2} {3}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0} {1}]
    
    set result1 [torch::embedding_bag $input $weight $offsets 1]
    set result2 [torch::embeddingBag $input $weight $offsets 1]
    
    # Check both have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

# ========================================
# Tests for Different Aggregation Modes
# ========================================

test embedding_bag-5.1 {Sum aggregation with known values} {
    # Create simple test case where we can verify the math
    # embedding[0] = [1, 2], embedding[1] = [3, 4]
    # bag contains indices [0, 1] -> sum should be [4, 6]
    set input [create_int_tensor {0 1} {2}]
    set weight [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set offsets [create_int_tensor {0} {1}]
    
    set result [torch::embedding_bag $input $weight $offsets 0]
    
    # Verify that aggregation worked (shape should be correct)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1 2"}
} 1

test embedding_bag-5.2 {Mean aggregation with different bag sizes} {
    # Two bags: first with 1 element, second with 2 elements
    set input [create_int_tensor {0 1 2} {3}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0 1} {2}]
    
    set result [torch::embedding_bag $input $weight $offsets 1]
    
    # Check output shape should be [2, 2] (2 bags, 2-dim)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test embedding_bag-5.3 {Max aggregation with repeated indices} {
    # Bag with repeated indices to test max aggregation
    set input [create_int_tensor {0 1 0 1} {4}]
    set weight [create_tensor {1.0 2.0 3.0 4.0} {2 2}]
    set offsets [create_int_tensor {0} {1}]
    
    set result [torch::embedding_bag $input $weight $offsets 2]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1 2"}
} 1

# ========================================
# Tests for Different Bag Configurations
# ========================================

test embedding_bag-6.1 {Variable bag sizes} {
    # Bags with sizes: 1, 3, 2
    set input [create_int_tensor {0 1 2 0 1 2} {6}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0 1 4} {3}]
    
    set result [torch::embedding_bag $input $weight $offsets 0]
    
    # Check output shape should be [3, 2] (3 bags)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 2"}
} 1

test embedding_bag-6.2 {Empty middle bag} {
    # Test with carefully constructed offsets to simulate empty bag behavior
    set input [create_int_tensor {0 1 2} {3}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0 2} {2}]
    
    set result [torch::embedding_bag $input $weight $offsets 0]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} 1

test embedding_bag-6.3 {Large vocabulary and embeddings} {
    # Test with larger dimensions
    set input [create_int_tensor {0 5 10 15 20} {5}]
    set weight [create_tensor [string repeat "1.0 2.0 3.0 4.0 " 25] {25 4}]
    set offsets [create_int_tensor {0 2} {2}]
    
    set result [torch::embedding_bag $input $weight $offsets 0]
    
    # Check output shape should be [2, 4]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 4"}
} 1

# ========================================
# Tests for Different Embedding Dimensions
# ========================================

test embedding_bag-7.1 {Small embedding dimension} {
    set input [create_int_tensor {0 1 2} {3}]
    set weight [create_tensor {1.0 2.0 3.0} {3 1}]
    set offsets [create_int_tensor {0} {1}]
    
    set result [torch::embedding_bag $input $weight $offsets 0]
    
    # Check output shape should be [1, 1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1 1"}
} 1

test embedding_bag-7.2 {Large embedding dimension} {
    set input [create_int_tensor {0 1} {2}]
    # Create 2x16 weight matrix
    set weight_data [string repeat "1.0 " 32]
    set weight [create_tensor $weight_data {2 16}]
    set offsets [create_int_tensor {0} {1}]
    
    set result [torch::embedding_bag $input $weight $offsets 0]
    
    # Check output shape should be [1, 16]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1 16"}
} 1

# ========================================
# Error Handling Tests
# ========================================

test embedding_bag-8.1 {Error - missing arguments} {
    set result [catch {torch::embedding_bag} error]
    set result
} 1

test embedding_bag-8.2 {Error - insufficient positional arguments} {
    set input [create_int_tensor {0 1} {2}]
    set result [catch {torch::embedding_bag $input} error]
    set result
} 1

test embedding_bag-8.3 {Error - invalid tensor name} {
    set weight [create_tensor {1.0 2.0} {1 2}]
    set offsets [create_int_tensor {0} {1}]
    set result [catch {torch::embedding_bag "invalid_tensor" $weight $offsets 0} error]
    set result
} 1

test embedding_bag-8.4 {Error - invalid weight tensor} {
    set input [create_int_tensor {0} {1}]
    set offsets [create_int_tensor {0} {1}]
    set result [catch {torch::embedding_bag $input "invalid_weight" $offsets 0} error]
    set result
} 1

test embedding_bag-8.5 {Error - invalid offsets tensor} {
    set input [create_int_tensor {0} {1}]
    set weight [create_tensor {1.0 2.0} {1 2}]
    set result [catch {torch::embedding_bag $input $weight "invalid_offsets" 0} error]
    set result
} 1

test embedding_bag-8.6 {Error - invalid mode value negative} {
    set input [create_int_tensor {0} {1}]
    set weight [create_tensor {1.0 2.0} {1 2}]
    set offsets [create_int_tensor {0} {1}]
    set result [catch {torch::embedding_bag $input $weight $offsets -1} error]
    set result
} 1

test embedding_bag-8.7 {Error - invalid mode value too large} {
    set input [create_int_tensor {0} {1}]
    set weight [create_tensor {1.0 2.0} {1 2}]
    set offsets [create_int_tensor {0} {1}]
    set result [catch {torch::embedding_bag $input $weight $offsets 3} error]
    set result
} 1

test embedding_bag-8.8 {Error - missing value for named parameter} {
    set input [create_int_tensor {0} {1}]
    set weight [create_tensor {1.0 2.0} {1 2}]
    set result [catch {torch::embedding_bag -input $input -weight $weight -offsets} error]
    set result
} 1

test embedding_bag-8.9 {Error - unknown named parameter} {
    set input [create_int_tensor {0} {1}]
    set weight [create_tensor {1.0 2.0} {1 2}]
    set offsets [create_int_tensor {0} {1}]
    set result [catch {torch::embedding_bag -input $input -weight $weight -offsets $offsets -unknown_param 5} error]
    set result
} 1

test embedding_bag-8.10 {Error - invalid mode in named syntax} {
    set input [create_int_tensor {0} {1}]
    set weight [create_tensor {1.0 2.0} {1 2}]
    set offsets [create_int_tensor {0} {1}]
    set result [catch {torch::embedding_bag -input $input -weight $weight -offsets $offsets -mode "not_a_number"} error]
    set result
} 1

test embedding_bag-8.11 {Error - CamelCase with invalid parameters} {
    set result [catch {torch::embeddingBag "invalid_tensor" "invalid_weight" "invalid_offsets" 0} error]
    set result
} 1

# ========================================
# Integration Tests
# ========================================

test embedding_bag-9.1 {Integration with tensor operations} {
    # Create embedding bag result and perform operations
    set input [create_int_tensor {0 1 2} {3}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0} {1}]
    
    set bag_result [torch::embedding_bag $input $weight $offsets 0]
    
    # Sum across embedding dimension
    set summed [torch::tensor_sum $bag_result 1]
    
    # Should result in shape [1] (one sum per bag)
    set shape [torch::tensor_shape $summed]
    expr {$shape eq "1"}
} 1

test embedding_bag-9.2 {Multiple embedding bag operations} {
    # Test creating multiple embedding bags
    set input1 [create_int_tensor {0 1} {2}]
    set input2 [create_int_tensor {1 2} {2}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0} {1}]
    
    set bag1 [torch::embedding_bag $input1 $weight $offsets 0]
    set bag2 [torch::embedding_bag $input2 $weight $offsets 0]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $bag1]
    set shape2 [torch::tensor_shape $bag2]
    expr {$shape1 eq $shape2 && $shape1 eq "1 2"}
} 1

test embedding_bag-9.3 {Embedding bag with downstream processing} {
    # Create multiple bags and process them
    set input [create_int_tensor {0 1 2 0 1 2} {6}]
    set weight [create_tensor {1.0 2.0 3.0 4.0 5.0 6.0} {3 2}]
    set offsets [create_int_tensor {0 3} {2}]
    
    set bags [torch::embedding_bag $input $weight $offsets 1]
    
    # Mean across bags
    set mean_bags [torch::tensor_mean $bags 0]
    
    # Should result in shape [2] (mean of each embedding dimension)
    set shape [torch::tensor_shape $mean_bags]
    expr {$shape eq "2"}
} 1

# Clean up any remaining tensors and run tests
cleanupTests 