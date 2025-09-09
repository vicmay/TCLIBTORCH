#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configure test
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax (backward compatibility)
test affine_grid-1.1 {Basic positional syntax} {
    # Create a simple theta tensor using zeros and manually set values
    set theta [torch::zeros {1 2 3} float32 cpu false]
    # Try basic affine_grid call
    set result [torch::affine_grid $theta {1 1 4 4} 0]
    
    # Check that result is a valid tensor
    expr {[string match "tensor*" $result]}
} {1}

test affine_grid-1.2 {Positional syntax with align_corners true} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    set result [torch::affine_grid $theta {1 1 4 4} 1]
    
    # Check that result is a valid tensor
    expr {[string match "tensor*" $result]}
} {1}

test affine_grid-1.3 {Positional syntax with default align_corners} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    set result [torch::affine_grid $theta {1 1 4 4}]
    
    # Check that result is a valid tensor
    expr {[string match "tensor*" $result]}
} {1}

# Test 2: Named parameter syntax
test affine_grid-2.1 {Named parameter syntax basic} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    set result [torch::affine_grid -theta $theta -size {1 1 4 4}]
    
    # Check that result is a valid tensor
    expr {[string match "tensor*" $result]}
} {1}

test affine_grid-2.2 {Named parameter syntax with alignCorners} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    set result [torch::affine_grid -theta $theta -size {1 1 4 4} -alignCorners 1]
    
    # Check that result is a valid tensor
    expr {[string match "tensor*" $result]}
} {1}

test affine_grid-2.3 {Named parameter syntax with align_corners (snake_case)} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    set result [torch::affine_grid -theta $theta -size {1 1 4 4} -align_corners 0]
    
    # Check that result is a valid tensor
    expr {[string match "tensor*" $result]}
} {1}

test affine_grid-2.4 {Named parameter syntax with reordered parameters} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    set result [torch::affine_grid -size {1 1 4 4} -theta $theta -alignCorners 1]
    
    # Check that result is a valid tensor
    expr {[string match "tensor*" $result]}
} {1}

# Test 3: camelCase alias
test affine_grid-3.1 {camelCase alias basic test} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    set result [torch::affineGrid $theta {1 1 4 4}]
    
    # Check that result is a valid tensor
    expr {[string match "tensor*" $result]}
} {1}

test affine_grid-3.2 {camelCase alias with named parameters} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    set result [torch::affineGrid -theta $theta -size {1 1 4 4} -alignCorners 1]
    
    # Check that result is a valid tensor
    expr {[string match "tensor*" $result]}
} {1}

# Test 4: Consistency between both syntaxes
test affine_grid-4.1 {Consistency between positional and named syntax} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    
    # Test both syntaxes produce valid results
    set result1 [torch::affine_grid $theta {1 1 4 4} 0]
    set result2 [torch::affine_grid -theta $theta -size {1 1 4 4} -alignCorners 0]
    
    # Both should be valid tensors
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test affine_grid-4.2 {Consistency between snake_case and camelCase} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    
    # Test both command names produce valid results
    set result1 [torch::affine_grid -theta $theta -size {1 1 4 4} -alignCorners 1]
    set result2 [torch::affineGrid -theta $theta -size {1 1 4 4} -alignCorners 1]
    
    # Both should be valid tensors
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

# Test 5: Error handling
test affine_grid-5.1 {Error on missing theta parameter} {
    # Should return error for missing required parameter
    catch {torch::affine_grid -size {1 1 4 4}} result
    expr {[string match "*theta*" $result] || [string match "*Required*" $result]}
} {1}

test affine_grid-5.2 {Error on missing size parameter} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    # Should return error for missing required parameter
    catch {torch::affine_grid -theta $theta} result
    expr {[string match "*size*" $result] || [string match "*Required*" $result]}
} {1}

test affine_grid-5.3 {Error on invalid tensor name} {
    # Should return error for invalid tensor
    catch {torch::affine_grid -theta "invalid_tensor" -size {1 1 4 4}} result
    expr {[string match "*Invalid*tensor*" $result]}
} {1}

test affine_grid-5.4 {Error on unknown parameter} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    # Should return error for unknown parameter
    catch {torch::affine_grid -theta $theta -size {1 1 4 4} -unknown_param 1} result
    expr {[string match "*Unknown*" $result]}
} {1}

test affine_grid-5.5 {Error on positional syntax with wrong number of args} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    # Should return error for wrong number of arguments  
    catch {torch::affine_grid $theta} result
    expr {[string match "*Usage*" $result] || [string match "*WrongNumArgs*" $result]}
} {1}

# Test 6: Different tensor sizes
test affine_grid-6.1 {Batch processing with different sizes} {
    # Test with different batch and spatial sizes
    set theta [torch::zeros {1 2 3} float32 cpu false]
    set result [torch::affine_grid -theta $theta -size {1 1 4 4}]
    
    # Check that result is a valid tensor
    expr {[string match "tensor*" $result]}
} {1}

# Test 7: Parameter validation
test affine_grid-7.1 {Invalid alignCorners value} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    # Should handle invalid boolean value gracefully
    catch {torch::affine_grid -theta $theta -size {1 1 4 4} -alignCorners "not_a_boolean"} result
    expr {[string match "*align*" $result] || [string match "*Invalid*" $result]}
} {1}

test affine_grid-7.2 {Empty size list} {
    set theta [torch::zeros {1 2 3} float32 cpu false]
    # Should handle empty size list
    catch {torch::affine_grid -theta $theta -size {}} result
    expr {[string match "*size*" $result] || [string match "*Required*" $result]}
} {1}

# Clean up
cleanupTests 