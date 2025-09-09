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

# Test cases for positional syntax
test tensor-masked-fill-1.1 {Basic positional syntax} {
    # Create test tensor
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set mask [torch::tensor_create {1 0 1 0} bool cpu false]
    
    # Test masked fill
    set result [torch::tensor_masked_fill $tensor $mask 0.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {0.0 2.0 0.0 4.0}

test tensor-masked-fill-1.2 {Positional syntax with different value} {
    # Create test tensor
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set mask [torch::tensor_create {0 1 0 1} bool cpu false]
    
    # Test masked fill with value 5.0
    set result [torch::tensor_masked_fill $tensor $mask 5.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {1.0 5.0 3.0 5.0}

# Test cases for named parameter syntax
test tensor-masked-fill-2.1 {Named parameter syntax} {
    # Create test tensor
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set mask [torch::tensor_create {1 0 1 0} bool cpu false]
    
    # Test masked fill with named parameters
    set result [torch::tensor_masked_fill -tensor $tensor -mask $mask -value 0.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {0.0 2.0 0.0 4.0}

test tensor-masked-fill-2.2 {Named parameter syntax with different value} {
    # Create test tensor
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set mask [torch::tensor_create {0 1 0 1} bool cpu false]
    
    # Test masked fill with named parameters
    set result [torch::tensor_masked_fill -tensor $tensor -mask $mask -value 5.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {1.0 5.0 3.0 5.0}

# Test cases for camelCase alias
test tensor-masked-fill-3.1 {CamelCase alias positional syntax} {
    # Create test tensor
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set mask [torch::tensor_create {1 0 1 0} bool cpu false]
    
    # Test masked fill using camelCase alias
    set result [torch::tensorMaskedFill $tensor $mask 0.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {0.0 2.0 0.0 4.0}

test tensor-masked-fill-3.2 {CamelCase alias named parameter syntax} {
    # Create test tensor
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set mask [torch::tensor_create {0 1 0 1} bool cpu false]
    
    # Test masked fill using camelCase alias with named parameters
    set result [torch::tensorMaskedFill -tensor $tensor -mask $mask -value 5.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {1.0 5.0 3.0 5.0}

# Error handling tests
test tensor-masked-fill-4.1 {Error handling - missing tensor} {
    set mask [torch::tensor_create {1 0 1 0} bool cpu false]
    
    # Test with non-existent tensor
    set result [catch {torch::tensor_masked_fill nonexistent_tensor $mask 0.0} error]
    
    return [list $result $error]
} {1 {Tensor not found}}

test tensor-masked-fill-4.2 {Error handling - missing mask} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    
    # Test with non-existent mask
    set result [catch {torch::tensor_masked_fill $tensor nonexistent_mask 0.0} error]
    
    return [list $result $error]
} {1 {Tensor not found}}

test tensor-masked-fill-4.3 {Error handling - invalid value type} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set mask [torch::tensor_create {1 0 1 0} bool cpu false]
    
    # Test with invalid value (string instead of number)
    set result [catch {torch::tensor_masked_fill $tensor $mask "invalid"} error]
    
    return [list $result $error]
} {1 {Invalid value parameter}}

test tensor-masked-fill-4.4 {Error handling - insufficient arguments} {
    set result [catch {torch::tensor_masked_fill tensor1} error]
    return [list $result $error]
} {1 {Usage: torch::tensor_masked_fill tensor mask value | torch::tensor_masked_fill -tensor tensor -mask mask -value value}}

test tensor-masked-fill-4.5 {Error handling - unknown named parameter} {
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set mask [torch::tensor_create {1 0 1 0} bool cpu false]
    
    # Test with unknown parameter
    set result [catch {torch::tensor_masked_fill -tensor $tensor -mask $mask -value 0.0 -unknown param} error]
    
    return [list $result $error]
} {1 {Unknown parameter: -unknown. Valid parameters are: -tensor, -mask, -value}}

# Test with 2D tensor
test tensor-masked-fill-5.1 {2D tensor masked fill} {
    # Create 2D test tensor
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set mask [torch::tensor_create {{1 0} {0 1}} bool cpu false]
    
    # Test masked fill
    set result [torch::tensor_masked_fill $tensor $mask 0.0]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {0.0 2.0 3.0 0.0}

# Test with negative values
test tensor-masked-fill-6.1 {Negative value masked fill} {
    # Create test tensor
    set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
    set mask [torch::tensor_create {1 0 1 0} bool cpu false]
    
    # Test masked fill with negative value
    set result [torch::tensor_masked_fill $tensor $mask -1.5]
    
    # Get tensor data
    set data [torch::tensor_to_list $result]
    
    return $data
} {-1.5 2.0 -1.5 4.0}

cleanupTests 