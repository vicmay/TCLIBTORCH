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

# Helper function to create a test tensor
proc create_test_tensor {name shape data} {
    set tensor [torch::tensor_create -data $data -shape $shape]
    return $tensor
}

# Test positional syntax (backward compatibility)
test distributed_broadcast-1.1 {Basic positional syntax} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributed_broadcast $tensor]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test distributed_broadcast-1.2 {Positional syntax with root} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributed_broadcast $tensor 0]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test distributed_broadcast-1.3 {Positional syntax with non-zero root} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributed_broadcast $tensor 2]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

# Test named parameter syntax
test distributed_broadcast-2.1 {Named parameter syntax} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributed_broadcast -tensor $tensor]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test distributed_broadcast-2.2 {Named parameter syntax with root} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributed_broadcast -tensor $tensor -root 0]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test distributed_broadcast-2.3 {Named parameter syntax with non-zero root} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributed_broadcast -tensor $tensor -root 3]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test distributed_broadcast-2.4 {Named parameters in different order} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributed_broadcast -root 1 -tensor $tensor]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

# Test camelCase alias
test distributed_broadcast-3.1 {CamelCase alias basic} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributedBroadcast $tensor]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test distributed_broadcast-3.2 {CamelCase alias with root} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributedBroadcast $tensor 1]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test distributed_broadcast-3.3 {CamelCase alias with named parameters} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributedBroadcast -tensor $tensor -root 2]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

# Test data consistency (simulated broadcast behavior)
test distributed_broadcast-4.1 {Data consistency between syntaxes} {
    set tensor [create_test_tensor "test" {2 2} {5.0 6.0 7.0 8.0}]
    
    set result1 [torch::distributed_broadcast $tensor 0]
    set result2 [torch::distributed_broadcast -tensor $tensor -root 0]
    set result3 [torch::distributedBroadcast $tensor 0]
    
    # All should return valid tensor handles
    set valid1 [string match "tensor*" $result1]
    set valid2 [string match "tensor*" $result2]
    set valid3 [string match "tensor*" $result3]
    
    expr {$valid1 && $valid2 && $valid3}
} {1}

test distributed_broadcast-4.2 {Shape preservation} {
    set tensor [create_test_tensor "test" {3 2} {1.0 2.0 3.0 4.0 5.0 6.0}]
    set result [torch::distributed_broadcast $tensor]
    
    set original_shape [torch::tensor_shape $tensor]
    set result_shape [torch::tensor_shape $result]
    
    expr {$original_shape eq $result_shape}
} {1}

test distributed_broadcast-4.3 {Data type preservation} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributed_broadcast $tensor]
    
    set original_dtype [torch::tensor_dtype $tensor]
    set result_dtype [torch::tensor_dtype $result]
    
    expr {$original_dtype eq $result_dtype}
} {1}

# Test different root values
test distributed_broadcast-5.1 {Multiple root values} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    
    # Test different root values
    set result0 [torch::distributed_broadcast $tensor 0]
    set result1 [torch::distributed_broadcast $tensor 1]
    set result2 [torch::distributed_broadcast $tensor 2]
    
    # All should return valid tensor handles
    set valid0 [string match "tensor*" $result0]
    set valid1 [string match "tensor*" $result1]
    set valid2 [string match "tensor*" $result2]
    
    expr {$valid0 && $valid1 && $valid2}
} {1}

test distributed_broadcast-5.2 {Large root value} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributed_broadcast $tensor 100]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

# Test error handling
test distributed_broadcast-6.1 {Missing tensor argument} {
    catch {torch::distributed_broadcast} msg
    expr {[string match "*wrong # args*" $msg]}
} {1}

test distributed_broadcast-6.2 {Invalid tensor handle} {
    catch {torch::distributed_broadcast invalid_tensor} msg
    expr {[string match "*Tensor not found*" $msg]}
} {1}

test distributed_broadcast-6.3 {Invalid root type} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    catch {torch::distributed_broadcast $tensor not_a_number} msg
    expr {[string match "*root must be an integer*" $msg]}
} {1}

test distributed_broadcast-6.4 {Negative root value} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    catch {torch::distributed_broadcast $tensor -1} msg
    expr {[string match "*root must be >= 0*" $msg]}
} {1}

test distributed_broadcast-6.5 {Named parameter missing value} {
    catch {torch::distributed_broadcast -tensor} msg
    expr {[string match "*wrong # args*" $msg]}
} {1}

test distributed_broadcast-6.6 {Unknown named parameter} {
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    catch {torch::distributed_broadcast -tensor $tensor -unknown value} msg
    expr {[string match "*unknown option*" $msg]}
} {1}

test distributed_broadcast-6.7 {Missing tensor in named syntax} {
    catch {torch::distributed_broadcast -root 0} msg
    expr {[string match "*tensor required*" $msg]}
} {1}

test distributed_broadcast-6.8 {CamelCase error handling} {
    catch {torch::distributedBroadcast} msg
    expr {[string match "*wrong # args*" $msg]}
} {1}

# Test different tensor types and shapes
test distributed_broadcast-7.1 {1D tensor} {
    set tensor [create_test_tensor "test" {4} {1.0 2.0 3.0 4.0}]
    set result [torch::distributed_broadcast $tensor]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test distributed_broadcast-7.2 {3D tensor} {
    set tensor [create_test_tensor "test" {2 2 2} {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0}]
    set result [torch::distributed_broadcast $tensor]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

test distributed_broadcast-7.3 {Scalar tensor} {
    set tensor [create_test_tensor "test" {} {42.0}]
    set result [torch::distributed_broadcast $tensor]
    
    # Should return a tensor handle
    expr {[string match "tensor*" $result]}
} {1}

# Test distributed initialization context
test distributed_broadcast-8.1 {Single GPU mode} {
    torch::distributed_init 0 1 "gloo"
    
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributed_broadcast $tensor]
    
    # Should work in single GPU mode
    expr {[string match "tensor*" $result]}
} {1}

test distributed_broadcast-8.2 {Multi-GPU simulation mode} {
    torch::distributed_init 0 4 "nccl"
    
    set tensor [create_test_tensor "test" {2 2} {1.0 2.0 3.0 4.0}]
    set result [torch::distributed_broadcast $tensor]
    
    # Should work in multi-GPU mode
    expr {[string match "tensor*" $result]}
} {1}

cleanupTests 