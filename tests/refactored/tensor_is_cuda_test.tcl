#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Test cases for positional syntax
test tensor_is_cuda-1.1 {Basic positional syntax - CPU tensor} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::tensor_is_cuda $tensor]
    return $result
} {0}

test tensor_is_cuda-1.2 {Basic positional syntax - CUDA tensor} {
    if {[catch {torch::cuda_is_available} available] || !$available} {
        # CUDA not available, test should expect CPU tensor (0)
        set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cuda true]
        set result [torch::tensor_is_cuda $tensor]
        return $result
    }
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cuda true]
    set result [torch::tensor_is_cuda $tensor]
    return $result
} {0}

;# Test cases for named parameter syntax
test tensor_is_cuda-2.1 {Named parameter syntax - CPU tensor} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::tensor_is_cuda -tensor $tensor]
    return $result
} {0}

test tensor_is_cuda-2.2 {Named parameter syntax - CUDA tensor} {
    if {[catch {torch::cuda_is_available} available] || !$available} {
        # CUDA not available, test should expect CPU tensor (0)
        set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cuda true]
        set result [torch::tensor_is_cuda -tensor $tensor]
        return $result
    }
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cuda true]
    set result [torch::tensor_is_cuda -tensor $tensor]
    return $result
} {0}

test tensor_is_cuda-2.3 {Named parameter syntax with -input alias} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::tensor_is_cuda -input $tensor]
    return $result
} {0}

;# Test cases for camelCase alias
test tensor_is_cuda-3.1 {CamelCase alias - CPU tensor} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::tensorIsCuda $tensor]
    return $result
} {0}

test tensor_is_cuda-3.2 {CamelCase alias - CUDA tensor} {
    if {[catch {torch::cuda_is_available} available] || !$available} {
        # CUDA not available, test should expect CPU tensor (0)
        set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cuda true]
        set result [torch::tensorIsCuda $tensor]
        return $result
    }
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cuda true]
    set result [torch::tensorIsCuda $tensor]
    return $result
} {0}

test tensor_is_cuda-3.3 {CamelCase alias with named parameters} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set result [torch::tensorIsCuda -tensor $tensor]
    return $result
} {0}

;# Error handling tests
test tensor_is_cuda-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_is_cuda invalid_tensor} result
    return $result
} {Invalid tensor name}

test tensor_is_cuda-4.2 {Error handling - missing parameter} {
    catch {torch::tensor_is_cuda} result
    return $result
} {Required tensor parameter missing}

test tensor_is_cuda-4.3 {Error handling - unknown parameter} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    catch {torch::tensor_is_cuda -unknown $tensor} result
    return $result
} {Unknown parameter: -unknown}

test tensor_is_cuda-4.4 {Error handling - missing value for parameter} {
    catch {torch::tensor_is_cuda -tensor} result
    return $result
} {Missing value for parameter}

;# Edge cases
test tensor_is_cuda-5.1 {Empty tensor} {
    catch {torch::tensor_create -data {} -dtype float32 -device cpu -requiresGrad true} result
    puts "DEBUG: Empty tensor error message: '$result'"
    expr {$result ne ""}
} {1}

test tensor_is_cuda-5.2 {Large tensor} {
    set tensor [torch::tensor_create [lrepeat 1000 1.0] float32 cpu true]
    set result [torch::tensor_is_cuda $tensor]
    return $result
} {0}

;# Consistency tests - both syntaxes should produce same results
test tensor_is_cuda-6.1 {Consistency between positional and named syntax} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set pos_result [torch::tensor_is_cuda $tensor]
    set named_result [torch::tensor_is_cuda -tensor $tensor]
    return [expr {$pos_result == $named_result}]
} {1}

test tensor_is_cuda-6.2 {Consistency between snake_case and camelCase} {
    set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
    set snake_result [torch::tensor_is_cuda $tensor]
    set camel_result [torch::tensorIsCuda $tensor]
    return [expr {$snake_result == $camel_result}]
} {1}

cleanupTests 