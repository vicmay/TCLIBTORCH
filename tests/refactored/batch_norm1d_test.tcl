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

# -----------------------------------------------------------------------------
# Helper to create random input tensor of shape (N, C, L)
proc _rand1d {batch channels length} {
    set total [expr {$batch * $channels * $length}]
    set data {}
    for {set i 0} {$i < $total} {incr i} {
        lappend data [expr {double(rand())}]
    }
    return [torch::tensor_create $data float32 cpu false]
}

# -----------------------------------------------------------------------------
# 1. Positional syntax (back-compat)
# -----------------------------------------------------------------------------
set inputTensor [_rand1d 2 8 5]
set inputTensor [torch::tensor_reshape $inputTensor {2 8 5}]

test batch_norm1d-1.1 {Create layer using positional syntax} {
    set layer [torch::batch_norm1d 8]
    expr {[string length $layer] > 0}
} {1}

# -----------------------------------------------------------------------------
# 2. Named parameter syntax
# -----------------------------------------------------------------------------

test batch_norm1d-2.1 {Create layer using named parameters} {
    set layer [torch::batch_norm1d -numFeatures 8 -eps 1e-5 -momentum 0.2 -affine true -trackRunningStats false]
    expr {[string length $layer] > 0}
} {1}

# -----------------------------------------------------------------------------
# 3. camelCase alias
# -----------------------------------------------------------------------------

test batch_norm1d-3.1 {Create layer using camelCase alias} {
    set layer [torch::batchNorm1d -numFeatures 8]
    expr {[string length $layer] > 0}
} {1}

# -----------------------------------------------------------------------------
# 4. Forward pass consistency
# -----------------------------------------------------------------------------

# test batch_norm1d-4.1 {Forward pass works for both syntaxes} {
#     set layerPos [torch::batch_norm1d 8]
#     set layerNamed [torch::batch_norm1d -numFeatures 8]
#     set outPos   [torch::layer_forward $layerPos  $::inputTensor]
#     set outNamed [torch::layer_forward $layerNamed $::inputTensor]
#     # Ensure shapes are identical
#     string equal [torch::tensor_shape $outPos] [torch::tensor_shape $outNamed]
# } {1}

# -----------------------------------------------------------------------------
# 5. Error handling
# -----------------------------------------------------------------------------

test batch_norm1d-5.1 {Missing required argument (named)} {
    set rc [catch {torch::batch_norm1d -eps 1e-5} err]
    expr {$rc == 1 && [string match "*numFeatures*" $err]}
} {1}

test batch_norm1d-5.2 {Unknown parameter} {
    set rc [catch {torch::batch_norm1d -numFeatures 8 -unknown 1} err]
    expr {$rc == 1 && [string match "*Unknown parameter*" $err]}
} {1}

cleanupTests 