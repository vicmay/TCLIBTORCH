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

# Helper function to create test triplet tensors
proc create_triplet_tensors {} {
    # Create anchor tensor
    set anchor [torch::tensorCreate -data {1.0 0.5 -0.2 0.8 -0.3 1.2} -shape {2 3}]
    # Create positive tensor
    set positive [torch::tensorCreate -data {0.9 0.6 -0.1 0.7 -0.2 1.1} -shape {2 3}]
    # Create negative tensor
    set negative [torch::tensorCreate -data {-0.8 1.5 0.9 -1.2 0.8 -0.5} -shape {2 3}]
    
    return [list $anchor $positive $negative]
}

# Test 1: Basic positional syntax
test triplet_margin_with_distance_loss-1.1 {Basic positional syntax} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_with_distance_loss $anchor $positive $negative]
    expr {$result ne ""}
} 1

# Test 2: Named parameter syntax
test triplet_margin_with_distance_loss-2.1 {Named parameter syntax basic} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative]
    expr {$result ne ""}
} 1

# Test 3: camelCase alias
test triplet_margin_with_distance_loss-3.1 {camelCase alias basic} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::tripletMarginWithDistanceLoss -anchor $anchor -positive $positive -negative $negative]
    expr {$result ne ""}
} 1

# Test 4: Different distance functions
test triplet_margin_with_distance_loss-4.1 {Distance function - cosine} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative -distanceFunction "cosine"]
    expr {$result ne ""}
} 1

test triplet_margin_with_distance_loss-4.2 {Distance function - pairwise} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative -distanceFunction "pairwise"]
    expr {$result ne ""}
} 1

test triplet_margin_with_distance_loss-4.3 {Distance function - euclidean} {
    set tensors [create_triplet_tensors]
    set anchor [lindex $tensors 0]
    set positive [lindex $tensors 1]
    set negative [lindex $tensors 2]
    
    set result [torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative -distanceFunction "euclidean"]
    expr {$result ne ""}
} 1

cleanupTests 