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
test tensor_numel-1.1 {Basic positional syntax - scalar tensor} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    set numel [torch::tensor_numel $tensor]
    return $numel
} {1}

test tensor_numel-1.2 {Basic positional syntax - 1D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set numel [torch::tensor_numel $tensor]
    return $numel
} {3}

test tensor_numel-1.3 {Basic positional syntax - 2D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set numel [torch::tensor_numel $reshaped]
    return $numel
} {4}

test tensor_numel-1.4 {Basic positional syntax - 3D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2 2}]
    set numel [torch::tensor_numel $reshaped]
    return $numel
} {8}

# Test cases for named syntax
test tensor_numel-2.1 {Named parameter syntax - scalar tensor} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    set numel [torch::tensor_numel -tensor $tensor]
    return $numel
} {1}

test tensor_numel-2.2 {Named parameter syntax - 1D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set numel [torch::tensor_numel -tensor $tensor]
    return $numel
} {3}

test tensor_numel-2.3 {Named parameter syntax - 2D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set numel [torch::tensor_numel -tensor $reshaped]
    return $numel
} {4}

test tensor_numel-2.4 {Named parameter syntax with -input parameter} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set numel [torch::tensor_numel -input $tensor]
    return $numel
} {3}

# Test cases for camelCase alias
test tensor_numel-3.1 {CamelCase alias - scalar tensor} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    set numel [torch::tensorNumel $tensor]
    return $numel
} {1}

test tensor_numel-3.2 {CamelCase alias - 1D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set numel [torch::tensorNumel $tensor]
    return $numel
} {3}

test tensor_numel-3.3 {CamelCase alias - 2D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set numel [torch::tensorNumel $reshaped]
    return $numel
} {4}

# Test cases for different data types
test tensor_numel-4.1 {Different data types - int32} {
    set tensor [torch::tensor_create -data {1 2 3 4} -dtype int32]
    set numel [torch::tensor_numel $tensor]
    return $numel
} {4}

test tensor_numel-4.2 {Different data types - float64} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64]
    set numel [torch::tensor_numel $tensor]
    return $numel
} {3}

# Test cases for different devices
test tensor_numel-5.1 {Different devices - CPU} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
    set numel [torch::tensor_numel $tensor]
    return $numel
} {3}

test tensor_numel-5.2 {Different devices - CUDA if available} {
    if {[torch::cuda_is_available]} {
        set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cuda]
        set numel [torch::tensor_numel $tensor]
        return $numel
    } else {
        return "CUDA not available"
    }
} {3}

# Test cases for edge cases
test tensor_numel-6.1 {Edge case - empty tensor} {
    set tensor [torch::tensor_create -data {} -dtype float32]
    set numel [torch::tensor_numel $tensor]
    return $numel
} {0}

test tensor_numel-6.2 {Edge case - large tensor} {
    set data {}
    for {set i 0} {$i < 100} {incr i} {
        lappend data $i.0
    }
    set tensor [torch::tensor_create -data $data -dtype float32]
    set numel [torch::tensor_numel $tensor]
    return $numel
} {100}

test tensor_numel-6.3 {Edge case - zero tensor} {
    set tensor [torch::tensor_create -data 0.0 -dtype float32]
    set numel [torch::tensor_numel $tensor]
    return $numel
} {1}

# Test cases for syntax consistency
test tensor_numel-7.1 {Syntax consistency - positional vs named} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set numel1 [torch::tensor_numel $reshaped]
    set numel2 [torch::tensor_numel -tensor $reshaped]
    return [expr {$numel1 eq $numel2}]
} {1}

test tensor_numel-7.2 {Syntax consistency - named vs camelCase} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set numel1 [torch::tensor_numel -tensor $reshaped]
    set numel2 [torch::tensorNumel $reshaped]
    return [expr {$numel1 eq $numel2}]
} {1}

# Error handling tests
test tensor_numel-8.1 {Error handling - missing tensor} {
    catch {torch::tensor_numel} result
    return $result
} {Required parameter missing: tensor}

test tensor_numel-8.2 {Error handling - invalid tensor name} {
    catch {torch::tensor_numel invalid_tensor} result
    return $result
} {Invalid tensor name: invalid_tensor}

test tensor_numel-8.3 {Error handling - missing parameter value} {
    catch {torch::tensor_numel -tensor} result
    return $result
} {Missing value for parameter}

test tensor_numel-8.4 {Error handling - unknown parameter} {
    set tensor [torch::tensor_create -data {1.0 2.0} -dtype float32]
    catch {torch::tensor_numel -unknown $tensor} result
    return $result
} {Unknown parameter: -unknown}

test tensor_numel-8.5 {Error handling - too many arguments} {
    set tensor [torch::tensor_create -data {1.0 2.0} -dtype float32]
    catch {torch::tensor_numel $tensor extra_arg} result
    return $result
} {Usage: torch::tensor_numel tensor}

# Test cases for mathematical correctness
test tensor_numel-9.1 {Mathematical correctness - shape matches numel} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0}
    set tensor [torch::tensor_create -data $data -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 3}]
    set numel [torch::tensor_numel $reshaped]
    return $numel
} {6}

test tensor_numel-9.2 {Mathematical correctness - large nested structure} {
    set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0}
    set tensor [torch::tensor_create -data $data -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {3 2 2}]
    set numel [torch::tensor_numel $reshaped]
    return $numel
} {12}

# Test cases for tensor operations consistency
test tensor_numel-10.1 {Tensor operations consistency - after reshape} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set numel [torch::tensor_numel $reshaped]
    return $numel
} {4}

test tensor_numel-10.2 {Tensor operations consistency - after permute} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set permuted [torch::tensor_permute -input $reshaped -dims {1 0}]
    set numel [torch::tensor_numel $permuted]
    return $numel
} {4}

# Test cases for relationship with tensor_shape
test tensor_numel-11.1 {Relationship with tensor_shape - 1D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set shape [torch::tensor_shape $tensor]
    set numel [torch::tensor_numel $tensor]
    # For 1D tensor, numel should equal the first (and only) dimension
    return [expr {$numel == [lindex $shape 0]}]
} {1}

test tensor_numel-11.2 {Relationship with tensor_shape - 2D tensor} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {2 2}]
    set shape [torch::tensor_shape $reshaped]
    set numel [torch::tensor_numel $reshaped]
    # For 2D tensor, numel should equal product of dimensions
    set expected [expr {[lindex $shape 0] * [lindex $shape 1]}]
    return [expr {$numel == $expected}]
} {1}

cleanupTests 