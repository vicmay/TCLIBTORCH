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
test tensor_shape-1.1 {Basic positional syntax - scalar tensor} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    set shape [torch::tensor_shape $tensor]
    return $shape
} {1}

test tensor_shape-1.2 {Basic positional syntax - 1D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set shape [torch::tensor_shape $tensor]
    return $shape
} {3}

test tensor_shape-1.3 {Basic positional syntax - 2D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set shape [torch::tensor_shape $reshaped]
    return $shape
} {2 2}

test tensor_shape-1.4 {Basic positional syntax - 3D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2 2}]
    set shape [torch::tensor_shape $reshaped]
    return $shape
} {2 2 2}

# Test cases for named syntax
test tensor_shape-2.1 {Named parameter syntax - scalar tensor} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    set shape [torch::tensor_shape -tensor $tensor]
    return $shape
} {1}

test tensor_shape-2.2 {Named parameter syntax - 1D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set shape [torch::tensor_shape -tensor $tensor]
    return $shape
} {3}

test tensor_shape-2.3 {Named parameter syntax - 2D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set shape [torch::tensor_shape -tensor $reshaped]
    return $shape
} {2 2}

test tensor_shape-2.4 {Named parameter syntax with -input parameter} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set shape [torch::tensor_shape -input $tensor]
    return $shape
} {3}

# Test cases for camelCase alias
test tensor_shape-3.1 {CamelCase alias - scalar tensor} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    set shape [torch::tensorShape $tensor]
    return $shape
} {1}

test tensor_shape-3.2 {CamelCase alias - 1D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set shape [torch::tensorShape $tensor]
    return $shape
} {3}

test tensor_shape-3.3 {CamelCase alias - 2D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set shape [torch::tensorShape $reshaped]
    return $shape
} {2 2}

# Test cases for different data types
test tensor_shape-4.1 {Different data types - int32} {
    set tensor [torch::tensor_create -data {1 2 3 4} -dtype int32]
    set shape [torch::tensor_shape $tensor]
    return $shape
} {4}

test tensor_shape-4.2 {Different data types - float64} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64]
    set shape [torch::tensor_shape $tensor]
    return $shape
} {3}

# Test cases for different devices
test tensor_shape-5.1 {Different devices - CPU} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set shape [torch::tensor_shape $tensor]
    return $shape
} {3}

test tensor_shape-5.2 {Different devices - CUDA if available} {
    if {[torch::cuda_is_available]} {
        set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
        set shape [torch::tensor_shape $tensor]
        return $shape
    } else {
        return "CUDA not available"
    }
} {3}

# Test cases for edge cases
test tensor_shape-6.1 {Edge case - empty tensor} {
    set tensor [torch::tensor_create -data {} -dtype float32]
    set shape [torch::tensor_shape $tensor]
    return $shape
} {0}

test tensor_shape-6.2 {Edge case - large tensor} {
    set data {}
    for {set i 0} {$i < 100} {incr i} {
        lappend data $i.0
    }
    set tensor [torch::tensor_create -data $data -dtype float32]
    set shape [torch::tensor_shape $tensor]
    return $shape
} {100}

test tensor_shape-6.3 {Edge case - zero tensor} {
    set tensor [torch::tensor_create -data 0.0 -dtype float32]
    set shape [torch::tensor_shape $tensor]
    return $shape
} {1}

# Test cases for syntax consistency
test tensor_shape-7.1 {Syntax consistency - positional vs named} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set shape1 [torch::tensor_shape $reshaped]
    set shape2 [torch::tensor_shape -tensor $reshaped]
    return [expr {$shape1 eq $shape2}]
} {1}

test tensor_shape-7.2 {Syntax consistency - named vs camelCase} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set shape1 [torch::tensor_shape -tensor $reshaped]
    set shape2 [torch::tensorShape $reshaped]
    return [expr {$shape1 eq $shape2}]
} {1}

# Error handling tests
test tensor_shape-8.1 {Error handling - missing tensor} {
    catch {torch::tensor_shape} result
    return $result
} {Required parameter missing: tensor}

test tensor_shape-8.2 {Error handling - invalid tensor name} {
    catch {torch::tensor_shape invalid_tensor} result
    return $result
} {Invalid tensor name: invalid_tensor}

test tensor_shape-8.3 {Error handling - missing parameter value} {
    catch {torch::tensor_shape -tensor} result
    return $result
} {Missing value for parameter}

test tensor_shape-8.4 {Error handling - unknown parameter} {
    set tensor [torch::tensor_create -data {1.0 2.0} -dtype float32]
    catch {torch::tensor_shape -unknown $tensor} result
    return $result
} {Unknown parameter: -unknown}

test tensor_shape-8.5 {Error handling - too many arguments} {
    set tensor [torch::tensor_create -data {1.0 2.0} -dtype float32]
    catch {torch::tensor_shape $tensor extra_arg} result
    return $result
} {Usage: torch::tensor_shape tensor}

# Test cases for mathematical correctness
test tensor_shape-9.1 {Mathematical correctness - shape matches data} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0}
    set tensor [torch::tensor_create -data $data -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 3}]
    set shape [torch::tensor_shape $reshaped]
    return $shape
} {2 3}

test tensor_shape-9.2 {Mathematical correctness - nested structure} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0}
    set tensor [torch::tensor_create -data $data -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {3 2 2}]
    set shape [torch::tensor_shape $reshaped]
    return $shape
} {3 2 2}

# Test cases for tensor operations consistency
test tensor_shape-10.1 {Tensor operations consistency - after reshape} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set shape [torch::tensor_shape $reshaped]
    return $shape
} {2 2}

test tensor_shape-10.2 {Tensor operations consistency - after permute} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set permuted [torch::tensor_permute -input $reshaped -dims {1 0}]
    set shape [torch::tensor_shape $permuted]
    return $shape
} {2 2}

cleanupTests 