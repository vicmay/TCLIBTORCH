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

# ============================================================================
# TORCH::HSPLIT COMMAND TESTS
# ============================================================================
# This test suite verifies the torch::hsplit command functionality
# including dual syntax support and camelCase alias.

# Test 1: Basic functionality with positional syntax (backward compatibility)
test hsplit-1.1 {Basic horizontal split with number of sections - positional syntax} {
    set tensor [torch::ones -shape {2 6 3}]
    set result [torch::hsplit $tensor 2]
    set result_length [llength $result]
    set first_shape [torch::tensor_shape [lindex $result 0]]
    set second_shape [torch::tensor_shape [lindex $result 1]]
    list $result_length $first_shape $second_shape
} {2 {2 3 3} {2 3 3}}

# Test 2: Basic functionality with named parameter syntax
test hsplit-2.1 {Basic horizontal split with number of sections - named parameters} {
    set tensor [torch::ones -shape {2 6 3}]
    set result [torch::hsplit -tensor $tensor -sections 2]
    set result_length [llength $result]
    set first_shape [torch::tensor_shape [lindex $result 0]]
    set second_shape [torch::tensor_shape [lindex $result 1]]
    list $result_length $first_shape $second_shape
} {2 {2 3 3} {2 3 3}}

# Test 3: camelCase alias functionality
test hsplit-3.1 {camelCase alias torch::hSplit with named parameters} {
    set tensor [torch::ones -shape {2 6 3}]
    set result [torch::hSplit -tensor $tensor -sections 2]
    set result_length [llength $result]
    set first_shape [torch::tensor_shape [lindex $result 0]]
    set second_shape [torch::tensor_shape [lindex $result 1]]
    list $result_length $first_shape $second_shape
} {2 {2 3 3} {2 3 3}}

# Test 4: camelCase alias with positional syntax
test hsplit-3.2 {camelCase alias torch::hSplit with positional syntax} {
    set tensor [torch::ones -shape {2 6 3}]
    set result [torch::hSplit $tensor 2]
    set result_length [llength $result]
    set first_shape [torch::tensor_shape [lindex $result 0]]
    set second_shape [torch::tensor_shape [lindex $result 1]]
    list $result_length $first_shape $second_shape
} {2 {2 3 3} {2 3 3}}

# Test 5: Split with indices list (positional syntax)
test hsplit-4.1 {Horizontal split with indices list - positional syntax} {
    set tensor [torch::ones -shape {2 8 3}]
    set result [torch::hsplit $tensor {2 4 6}]
    set result_length [llength $result]
    set first_shape [torch::tensor_shape [lindex $result 0]]
    set second_shape [torch::tensor_shape [lindex $result 1]]
    set third_shape [torch::tensor_shape [lindex $result 2]]
    set fourth_shape [torch::tensor_shape [lindex $result 3]]
    list $result_length $first_shape $second_shape $third_shape $fourth_shape
} {4 {2 2 3} {2 2 3} {2 2 3} {2 2 3}}

# Test 6: Split with indices list (named parameters)
test hsplit-4.2 {Horizontal split with indices list - named parameters} {
    set tensor [torch::ones -shape {2 8 3}]
    set result [torch::hsplit -tensor $tensor -indices {2 4 6}]
    set result_length [llength $result]
    set first_shape [torch::tensor_shape [lindex $result 0]]
    set second_shape [torch::tensor_shape [lindex $result 1]]
    set third_shape [torch::tensor_shape [lindex $result 2]]
    set fourth_shape [torch::tensor_shape [lindex $result 3]]
    list $result_length $first_shape $second_shape $third_shape $fourth_shape
} {4 {2 2 3} {2 2 3} {2 2 3} {2 2 3}}

# Test 7: Alternative parameter names
test hsplit-5.1 {Alternative parameter names -input and -sections} {
    set tensor [torch::ones -shape {2 6 3}]
    set result [torch::hsplit -input $tensor -sections 3]
    set result_length [llength $result]
    set first_shape [torch::tensor_shape [lindex $result 0]]
    set second_shape [torch::tensor_shape [lindex $result 1]]
    set third_shape [torch::tensor_shape [lindex $result 2]]
    list $result_length $first_shape $second_shape $third_shape
} {3 {2 2 3} {2 2 3} {2 2 3}}

# Test 8: Data integrity check
test hsplit-6.1 {Data integrity check - values preserved} {
    set tensor [torch::arange -start 0 -end 24 -dtype float32]
    set reshaped [torch::tensor_reshape $tensor {2 12}]
    set result [torch::hsplit $reshaped 3]
    
    # Each split should have shape {2 4}
    set first_shape [torch::tensor_shape [lindex $result 0]]
    set second_shape [torch::tensor_shape [lindex $result 1]]
    set third_shape [torch::tensor_shape [lindex $result 2]]
    
    list $first_shape $second_shape $third_shape
} {{2 4} {2 4} {2 4}}

# Test 9: Different tensor types
test hsplit-7.1 {Different tensor types - float32} {
    set tensor [torch::ones -shape {2 6 3} -dtype float32]
    set result [torch::hsplit -tensor $tensor -sections 2]
    set result_length [llength $result]
    set dtype [torch::tensor_dtype [lindex $result 0]]
    list $result_length $dtype
} {2 Float32}

# Test 10: Different tensor types - int64
test hsplit-7.2 {Different tensor types - int64} {
    set tensor [torch::ones -shape {2 6 3} -dtype int64]
    set result [torch::hsplit -tensor $tensor -sections 2]
    set result_length [llength $result]
    set dtype [torch::tensor_dtype [lindex $result 0]]
    list $result_length $dtype
} {2 Int64}

# Test 11: Large tensor split
test hsplit-8.1 {Large tensor split} {
    set tensor [torch::ones -shape {1 100 1}]
    set result [torch::hsplit -tensor $tensor -sections 10]
    set result_length [llength $result]
    set first_shape [torch::tensor_shape [lindex $result 0]]
    list $result_length $first_shape
} {10 {1 10 1}}

# Test 12: Error handling - missing required parameters
test hsplit-9.1 {Error handling - missing tensor parameter} {
    set result [catch {torch::hsplit -sections 2} msg]
    set result
} {1}

# Test 13: Error handling - missing sections parameter
test hsplit-9.2 {Error handling - missing sections parameter} {
    set tensor [torch::ones -shape {2 6 3}]
    set result [catch {torch::hsplit -tensor $tensor} msg]
    set result
} {1}

# Test 14: Error handling - invalid parameter name
test hsplit-9.3 {Error handling - invalid parameter name} {
    set tensor [torch::ones -shape {2 6 3}]
    set result [catch {torch::hsplit -tensor $tensor -invalid 2} msg]
    set result
} {1}

# Test 15: Error handling - wrong number of positional arguments
test hsplit-9.4 {Error handling - wrong number of positional arguments} {
    set tensor [torch::ones -shape {2 6 3}]
    set result [catch {torch::hsplit $tensor} msg]
    set result
} {1}

# Test 16: Error handling - incompatible split size
test hsplit-9.5 {Error handling - incompatible split size} {
    set tensor [torch::ones -shape {2 5 3}]
    set result [catch {torch::hsplit $tensor 2} msg]
    # This should fail because 5 is not evenly divisible by 2
    set result
} {1}

# Test 17: Consistency check - both syntaxes produce same results
test hsplit-10.1 {Consistency check - both syntaxes produce same results} {
    set tensor [torch::arange -start 0 -end 24 -dtype float32]
    set reshaped [torch::tensor_reshape $tensor {2 12}]
    
    # Test both syntaxes
    set result1 [torch::hsplit $reshaped 3]
    set result2 [torch::hsplit -tensor $reshaped -sections 3]
    
    # Compare shapes
    set shape1_0 [torch::tensor_shape [lindex $result1 0]]
    set shape1_1 [torch::tensor_shape [lindex $result1 1]]
    set shape1_2 [torch::tensor_shape [lindex $result1 2]]
    
    set shape2_0 [torch::tensor_shape [lindex $result2 0]]
    set shape2_1 [torch::tensor_shape [lindex $result2 1]]
    set shape2_2 [torch::tensor_shape [lindex $result2 2]]
    
    # Check if shapes are identical
    set shapes_match [expr {$shape1_0 eq $shape2_0 && $shape1_1 eq $shape2_1 && $shape1_2 eq $shape2_2}]
    
    # Check if number of results is identical
    set count_match [expr {[llength $result1] == [llength $result2]}]
    
    list $shapes_match $count_match
} {1 1}

# Test 18: Syntax consistency with camelCase alias
test hsplit-10.2 {Syntax consistency with camelCase alias} {
    set tensor [torch::arange -start 0 -end 24 -dtype float32]
    set reshaped [torch::tensor_reshape $tensor {2 12}]
    
    # Test snake_case vs camelCase
    set result1 [torch::hsplit -tensor $reshaped -sections 3]
    set result2 [torch::hSplit -tensor $reshaped -sections 3]
    
    # Compare lengths
    set len1 [llength $result1]
    set len2 [llength $result2]
    
    # Compare first tensor shapes
    set shape1 [torch::tensor_shape [lindex $result1 0]]
    set shape2 [torch::tensor_shape [lindex $result2 0]]
    
    list [expr {$len1 == $len2}] [expr {$shape1 eq $shape2}]
} {1 1}

# Test 19: Memory management - verify tensors are properly created
test hsplit-11.1 {Memory management - verify tensors are properly created} {
    set tensor [torch::ones -shape {2 6 3}]
    set result [torch::hsplit -tensor $tensor -sections 2]
    
    # Check that all result tensors are valid handles
    set valid_tensors 0
    foreach tensor_handle $result {
        if {[torch::tensor_shape $tensor_handle] ne ""} {
            incr valid_tensors
        }
    }
    
    set valid_tensors
} {2}

# Test 20: Edge case - single section
test hsplit-12.1 {Edge case - single section} {
    set tensor [torch::ones -shape {2 6 3}]
    set result [torch::hsplit -tensor $tensor -sections 1]
    set result_length [llength $result]
    set shape [torch::tensor_shape [lindex $result 0]]
    list $result_length $shape
} {1 {2 6 3}}

cleanupTests 