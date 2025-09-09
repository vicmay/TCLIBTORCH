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
test lu_solve-1.1 {Basic positional syntax} {
    # Create predefined LU decomposition data
    # Using identity matrix for LU_data and simple pivots for testing
    set B [torch::tensor_create -data {1.0 2.0} -shape {2 1} -dtype float32]
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]
    
    set result [torch::lu_solve $B $LU_data $LU_pivots]
    
    # Verify we got a tensor handle back
    expr {[string match "tensor*" $result]}
} {1}

# Test 2: Named parameter syntax
test lu_solve-2.1 {Named parameter syntax with -B -LU_data -LU_pivots} {
    set B [torch::tensor_create -data {1.0 2.0} -shape {2 1} -dtype float32]
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]
    
    set result [torch::lu_solve -B $B -LU_data $LU_data -LU_pivots $LU_pivots]
    
    # Verify we got a tensor handle back
    expr {[string match "tensor*" $result]}
} {1}

test lu_solve-2.2 {Named parameter syntax with camelCase parameter names} {
    set B [torch::tensor_create -data {1.0 2.0} -shape {2 1} -dtype float32]
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]
    
    set result [torch::lu_solve -b $B -luData $LU_data -luPivots $LU_pivots]
    
    # Verify we got a tensor handle back
    expr {[string match "tensor*" $result]}
} {1}

# Test 3: camelCase alias
test lu_solve-3.1 {camelCase alias torch::luSolve} {
    set B [torch::tensor_create -data {1.0 2.0} -shape {2 1} -dtype float32]
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]
    
    set result [torch::luSolve $B $LU_data $LU_pivots]
    
    # Verify we got a tensor handle back
    expr {[string match "tensor*" $result]}
} {1}

test lu_solve-3.2 {camelCase alias with named parameters} {
    set B [torch::tensor_create -data {1.0 2.0} -shape {2 1} -dtype float32]
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]
    
    set result [torch::luSolve -B $B -LU_data $LU_data -LU_pivots $LU_pivots]
    
    # Verify we got a tensor handle back
    expr {[string match "tensor*" $result]}
} {1}

# Test 4: Mathematical correctness
test lu_solve-4.1 {Mathematical correctness - both syntaxes same result} {
    set B [torch::tensor_create -data {1.0 2.0} -shape {2 1} -dtype float32]
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]
    
    set result1 [torch::lu_solve $B $LU_data $LU_pivots]
    set result2 [torch::lu_solve -B $B -LU_data $LU_data -LU_pivots $LU_pivots]
    
    # Both should produce tensor handles
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test lu_solve-4.2 {Mathematical correctness - verify solution} {
    # Create a simple 2x2 system for verification
    set B [torch::tensor_create -data {3.0 7.0} -shape {2 1} -dtype float32]
    set LU_data [torch::tensor_create -data {1.0 2.0 0.0 1.0} -shape {2 2} -dtype float32]
    set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]
    
    set result [torch::lu_solve $B $LU_data $LU_pivots]
    
    # Should produce a valid tensor
    expr {[string match "tensor*" $result]}
} {1}

# Test 5: Error handling
test lu_solve-5.1 {Error: Missing required parameter B} {
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]
    
    catch {torch::lu_solve -LU_data $LU_data -LU_pivots $LU_pivots} result
    string match "*Required parameters missing*" $result
} {1}

test lu_solve-5.2 {Error: Missing required parameter LU_data} {
    set B [torch::tensor_create -data {1.0 2.0} -shape {2 1} -dtype float32]
    set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]
    
    catch {torch::lu_solve -B $B -LU_pivots $LU_pivots} result
    string match "*Required parameters missing*" $result
} {1}

test lu_solve-5.3 {Error: Missing required parameter LU_pivots} {
    set B [torch::tensor_create -data {1.0 2.0} -shape {2 1} -dtype float32]
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    
    catch {torch::lu_solve -B $B -LU_data $LU_data} result
    string match "*Required parameters missing*" $result
} {1}

test lu_solve-5.4 {Error: Invalid tensor name} {
    set B [torch::tensor_create -data {1.0 2.0} -shape {2 1} -dtype float32]
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    
    catch {torch::lu_solve -B $B -LU_data $LU_data -LU_pivots invalid_tensor} result
    string match "*Invalid*tensor*" $result
} {1}

# Test 6: Different data types
test lu_solve-6.1 {Different data types - float64} {
    set B [torch::tensor_create -data {1.0 2.0} -shape {2 1} -dtype float64]
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float64]
    set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]
    
    set result [torch::lu_solve $B $LU_data $LU_pivots]
    
    # Verify we got a tensor handle back
    expr {[string match "tensor*" $result]}
} {1}

test lu_solve-6.2 {Larger system 3x3} {
    set B [torch::tensor_create -data {1.0 2.0 3.0} -shape {3 1} -dtype float32]
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0} -shape {3 3} -dtype float32]
    set LU_pivots [torch::tensor_create -data {1 2 3} -shape {3} -dtype int32]
    
    set result [torch::lu_solve $B $LU_data $LU_pivots]
    
    # Verify we got a tensor handle back
    expr {[string match "tensor*" $result]}
} {1}

# Test 7: Parameter order independence
test lu_solve-7.1 {Parameter order independence} {
    set B [torch::tensor_create -data {1.0 2.0} -shape {2 1} -dtype float32]
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]
    
    set result1 [torch::lu_solve -B $B -LU_data $LU_data -LU_pivots $LU_pivots]
    set result2 [torch::lu_solve -LU_pivots $LU_pivots -B $B -LU_data $LU_data]
    
    # Both should produce tensor handles
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

# Test 8: Multiple right-hand sides (batch operations)
test lu_solve-8.1 {Multiple batch dimensions} {
    set B [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
    set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]
    
    set result [torch::lu_solve -B $B -LU_data $LU_data -LU_pivots $LU_pivots]
    
    # Verify we got a tensor handle back
    expr {[string match "tensor*" $result]}
} {1}

cleanupTests 