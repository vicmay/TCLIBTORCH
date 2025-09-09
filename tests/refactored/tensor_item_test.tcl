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
test tensor_item-1.1 {Basic positional syntax - float32 scalar} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    set value [torch::tensor_item $tensor]
    return $value
} {5.000000}

test tensor_item-1.2 {Basic positional syntax - int32 scalar} {
    set tensor [torch::tensor_create -data 42 -dtype int32]
    set value [torch::tensor_item $tensor]
    return $value
} {42}

test tensor_item-1.3 {Basic positional syntax - float64 scalar} {
    set tensor [torch::tensor_create -data 3.14159 -dtype float64]
    set value [torch::tensor_item $tensor]
    return $value
} {3.141590}

# Test cases for named syntax
test tensor_item-2.1 {Named parameter syntax - float32 scalar} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    set value [torch::tensor_item -tensor $tensor]
    return $value
} {5.000000}

test tensor_item-2.2 {Named parameter syntax - int32 scalar} {
    set tensor [torch::tensor_create -data 42 -dtype int32]
    set value [torch::tensor_item -tensor $tensor]
    return $value
} {42}

test tensor_item-2.3 {Named parameter syntax with -input parameter} {
    set tensor [torch::tensor_create -data 3.14159 -dtype float64]
    set value [torch::tensor_item -input $tensor]
    return $value
} {3.141590}

# Test cases for camelCase alias
test tensor_item-3.1 {CamelCase alias - float32 scalar} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    set value [torch::tensorItem $tensor]
    return $value
} {5.000000}

test tensor_item-3.2 {CamelCase alias - int32 scalar} {
    set tensor [torch::tensor_create -data 42 -dtype int32]
    set value [torch::tensorItem $tensor]
    return $value
} {42}

test tensor_item-3.3 {CamelCase alias with named parameters} {
    set tensor [torch::tensor_create -data 3.14159 -dtype float64]
    set value [torch::tensorItem -tensor $tensor]
    return $value
} {3.141590}

# Test cases for different data types
test tensor_item-4.1 {Different data types - int64} {
    set tensor [torch::tensor_create -data 123456789 -dtype int64]
    set value [torch::tensor_item $tensor]
    return $value
} {123456789}

test tensor_item-4.2 {Different data types - negative float} {
    set tensor [torch::tensor_create -data -2.5 -dtype float32]
    set value [torch::tensor_item $tensor]
    return $value
} {-2.500000}

test tensor_item-4.3 {Different data types - zero} {
    set tensor [torch::tensor_create -data 0.0 -dtype float32]
    set value [torch::tensor_item $tensor]
    return $value
} {0.000000}

# Test cases for different devices
test tensor_item-5.1 {Different devices - CPU} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32 -device cpu]
    set value [torch::tensor_item $tensor]
    return $value
} {5.000000}

test tensor_item-5.2 {Different devices - CUDA if available} {
    if {[torch::cuda_is_available]} {
        set tensor [torch::tensor_create -data 5.0 -dtype float32 -device cuda]
        set value [torch::tensor_item $tensor]
        return $value
    } else {
        return "CUDA not available"
    }
} {5.000000}

# Test cases for edge cases
test tensor_item-6.1 {Edge case - very small number} {
    set tensor [torch::tensor_create -data 0.000001 -dtype float32]
    set value [torch::tensor_item $tensor]
    return $value
} {0.000001}

test tensor_item-6.2 {Edge case - very large number} {
    set tensor [torch::tensor_create -data 999999.0 -dtype float32]
    set value [torch::tensor_item $tensor]
    return $value
} {999999.000000}

test tensor_item-6.3 {Edge case - negative integer} {
    set tensor [torch::tensor_create -data -100 -dtype int32]
    set value [torch::tensor_item $tensor]
    return $value
} {-100}

# Test cases for syntax consistency
test tensor_item-7.1 {Syntax consistency - positional vs named} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    set value1 [torch::tensor_item $tensor]
    set value2 [torch::tensor_item -tensor $tensor]
    return [expr {$value1 eq $value2}]
} {1}

test tensor_item-7.2 {Syntax consistency - named vs camelCase} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    set value1 [torch::tensor_item -tensor $tensor]
    set value2 [torch::tensorItem $tensor]
    return [expr {$value1 eq $value2}]
} {1}

# Error handling tests
test tensor_item-8.1 {Error handling - missing tensor} {
    catch {torch::tensor_item} result
    return $result
} {Required parameter missing: tensor}

test tensor_item-8.2 {Error handling - invalid tensor name} {
    catch {torch::tensor_item invalid_tensor} result
    return $result
} {Invalid tensor name: invalid_tensor}

test tensor_item-8.3 {Error handling - missing parameter value} {
    catch {torch::tensor_item -tensor} result
    return $result
} {Missing value for parameter}

test tensor_item-8.4 {Error handling - unknown parameter} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    catch {torch::tensor_item -unknown $tensor} result
    return $result
} {Unknown parameter: -unknown}

test tensor_item-8.5 {Error handling - too many arguments} {
    set tensor [torch::tensor_create -data 5.0 -dtype float32]
    catch {torch::tensor_item $tensor extra_arg} result
    return $result
} {Usage: torch::tensor_item tensor}

test tensor_item-8.6 {Error handling - tensor with multiple elements} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    catch {torch::tensor_item $tensor} result
    return $result
} {Tensor must have exactly one element}

test tensor_item-8.7 {Error handling - empty tensor} {
    set tensor [torch::tensor_create -data {} -dtype float32]
    catch {torch::tensor_item $tensor} result
    return $result
} {Tensor must have exactly one element}

# Test cases for mathematical correctness
test tensor_item-9.1 {Mathematical correctness - exact value preservation} {
    set tensor [torch::tensor_create -data 3.14159265359 -dtype float64]
    set value [torch::tensor_item $tensor]
    return $value
} {3.141593}

test tensor_item-9.2 {Mathematical correctness - integer precision} {
    set tensor [torch::tensor_create -data 2147483647 -dtype int32]
    set value [torch::tensor_item $tensor]
    return $value
} {2147483647}

# Test cases for tensor operations consistency
test tensor_item-10.1 {Tensor operations consistency - after reshape} {
    set tensor [torch::tensor_create -data {5.0} -dtype float32]
    set reshaped [torch::tensor_reshape -input $tensor -shape {1}]
    set value [torch::tensor_item $reshaped]
    return $value
} {5.000000}

test tensor_item-10.2 {Tensor operations consistency - after sum} {
    set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
    set summed [torch::tensor_sum -input $tensor]
    set value [torch::tensor_item $summed]
    return $value
} {6.000000}

cleanupTests 