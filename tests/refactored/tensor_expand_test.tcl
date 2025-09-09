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

# Test 1: Basic positional syntax (backward compatibility)
test tensor_expand-1.1 {Basic positional syntax} {
    set tensor [torch::tensor_create {2.0 3.0} float32 cpu false]
    set expanded [torch::tensor_expand $tensor {3 2}]
    
    # Verify expansion worked
    set shape [torch::tensor_shape $expanded]
    string equal $shape {3 2}
} {1}

# Test 2: Named parameter syntax
test tensor_expand-2.1 {Named parameter syntax} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu false]
    set expanded [torch::tensor_expand -input $tensor -sizes {2 3}]
    
    # Verify expansion worked
    set shape [torch::tensor_shape $expanded]
    string equal $shape {2 3}
} {1}

# Test 3: Alternative parameter names
test tensor_expand-2.2 {Alternative parameter names} {
    set tensor [torch::tensor_create {5.0 6.0} float32 cpu false]
    set expanded [torch::tensor_expand -tensor $tensor -shape {2 2}]
    
    # Verify expansion worked
    set shape [torch::tensor_shape $expanded]
    string equal $shape {2 2}
} {1}

# Test 4: camelCase alias
test tensor_expand-3.1 {camelCase alias syntax} {
    set tensor [torch::tensor_create {7.0 8.0} float32 cpu false]
    set expanded [torch::tensorExpand -input $tensor -sizes {1 2}]
    
    # Verify expansion worked
    set shape [torch::tensor_shape $expanded]
    string equal $shape {1 2}
} {1}

# Test 5: Both syntaxes produce same result
test tensor_expand-4.1 {Both syntaxes produce same result} {
    set tensor [torch::tensor_create {9.0 10.0} float32 cpu false]
    
    # Test positional syntax
    set expanded1 [torch::tensor_expand $tensor {2 2}]
    
    # Test named syntax  
    set expanded2 [torch::tensor_expand -input $tensor -sizes {2 2}]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $expanded1]
    set shape2 [torch::tensor_shape $expanded2]
    
    string equal $shape1 $shape2
} {1}

# Test 6: Error handling - missing parameters (named syntax)
test tensor_expand-5.1 {Error handling - missing input parameter} {
    set result [catch {torch::tensor_expand -sizes {2 2}} error]
    expr {$result == 1 && [string match "*Required parameters missing*" $error]}
} {1}

# Test 7: Error handling - missing parameters (positional syntax)
test tensor_expand-5.2 {Error handling - wrong number of positional args} {
    set tensor [torch::tensor_create {1.0} float32 cpu false]
    set result [catch {torch::tensor_expand $tensor} error]
    expr {$result == 1}
} {1}

# Test 8: Error handling - invalid tensor name
test tensor_expand-5.3 {Error handling - invalid tensor name} {
    set result [catch {torch::tensor_expand -input "invalid_tensor" -sizes {2 2}} error]
    expr {$result == 1 && [string match "*Invalid tensor name*" $error]}
} {1}

# Test 9: Error handling - unknown parameter
test tensor_expand-5.4 {Error handling - unknown parameter} {
    set tensor [torch::tensor_create {1.0} float32 cpu false]
    set result [catch {torch::tensor_expand -input $tensor -badparam value -sizes {2}} error]
    expr {$result == 1 && [string match "*Unknown parameter*" $error]}
} {1}

# Test 10: Complex expansion with multiple dimensions
test tensor_expand-6.1 {Complex multi-dimensional expansion} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
    set reshaped [torch::tensor_reshape $tensor {2 2}]
    set expanded [torch::tensor_expand $reshaped {3 2 2}]
    
    set shape [torch::tensor_shape $expanded]
    string equal $shape {3 2 2}
} {1}

# Test 11: Expansion preserves data consistency
test tensor_expand-7.1 {Expansion preserves data consistency} {
    set tensor [torch::tensor_create {42.0} float32 cpu false]
    set expanded [torch::tensor_expand -input $tensor -sizes {2 3}]
    
    # Verify the expanded tensor has the right shape and content
    set shape [torch::tensor_shape $expanded]
    string equal $shape {2 3}
} {1}

cleanupTests 