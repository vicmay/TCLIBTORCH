#!/usr/bin/env tclsh

package require tcltest
namespace import tcltest::*

# Load the shared library
if {[catch {load ../../build/libtorchtcl.so} result]} {
    puts "Failed to load libtorchtcl.so: $result"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip start}

# Helper to create test tensors
proc create_test_tensor {name shape} {
    set tensor [torch::zeros -shape $shape]
    return $tensor
}

# Test suite for torch::distributed_isend command
test distributed_isend-1.1 {Basic positional syntax} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle [torch::distributed_isend $tensor 1]
    regexp {isend_handle_dst1_tag0} $handle
} {1}

test distributed_isend-1.2 {Positional syntax with tag} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle [torch::distributed_isend $tensor 2 42]
    regexp {isend_handle_dst2_tag42} $handle
} {1}

test distributed_isend-1.3 {Positional syntax with different dst} {
    set tensor [create_test_tensor "test_tensor" {3 3}]
    set handle [torch::distributed_isend $tensor 5]
    regexp {isend_handle_dst5_tag0} $handle
} {1}

# Test named parameter syntax
test distributed_isend-2.1 {Named parameter syntax basic} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle [torch::distributed_isend -tensor $tensor -dst 1]
    regexp {isend_handle_dst1_tag0} $handle
} {1}

test distributed_isend-2.2 {Named parameter syntax with tag} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle [torch::distributed_isend -tensor $tensor -dst 3 -tag 100]
    regexp {isend_handle_dst3_tag100} $handle
} {1}

test distributed_isend-2.3 {Named parameter syntax different order} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle [torch::distributed_isend -dst 2 -tensor $tensor -tag 25]
    regexp {isend_handle_dst2_tag25} $handle
} {1}

# Test camelCase alias
test distributed_isend-3.1 {camelCase alias basic} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle [torch::distributedIsend $tensor 1]
    regexp {isend_handle_dst1_tag0} $handle
} {1}

test distributed_isend-3.2 {camelCase alias with named parameters} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle [torch::distributedIsend -tensor $tensor -dst 4 -tag 50]
    regexp {isend_handle_dst4_tag50} $handle
} {1}

test distributed_isend-3.3 {camelCase alias with tag} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle [torch::distributedIsend $tensor 3 15]
    regexp {isend_handle_dst3_tag15} $handle
} {1}

# Test command existence
test distributed_isend-4.1 {Verify torch::distributed_isend command exists} {
    info commands torch::distributed_isend
} {::torch::distributed_isend}

test distributed_isend-4.2 {Verify torch::distributedIsend camelCase alias exists} {
    info commands torch::distributedIsend
} {::torch::distributedIsend}

# Test different tensor shapes
test distributed_isend-5.1 {1D tensor} {
    set tensor [create_test_tensor "test_tensor" {10}]
    set handle [torch::distributed_isend $tensor 1]
    regexp {isend_handle_dst1_tag0} $handle
} {1}

test distributed_isend-5.2 {3D tensor} {
    set tensor [create_test_tensor "test_tensor" {2 3 4}]
    set handle [torch::distributed_isend $tensor 2]
    regexp {isend_handle_dst2_tag0} $handle
} {1}

test distributed_isend-5.3 {4D tensor} {
    set tensor [create_test_tensor "test_tensor" {1 2 3 4}]
    set handle [torch::distributed_isend $tensor 0]
    regexp {isend_handle_dst0_tag0} $handle
} {1}

# Error handling tests
test distributed_isend-6.1 {Missing arguments} {
    catch {torch::distributed_isend} error
    string match "*Required parameters missing*" $error
} {1}

test distributed_isend-6.2 {Missing dst in positional syntax} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    catch {torch::distributed_isend $tensor} error
    string match "*Wrong number of arguments*" $error
} {1}

test distributed_isend-6.3 {Invalid tensor handle} {
    catch {torch::distributed_isend "invalid_tensor" 1} error
    string match "*Invalid tensor handle*" $error
} {1}

test distributed_isend-6.4 {Invalid dst parameter} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    catch {torch::distributed_isend $tensor "invalid_dst"} error
    string match "*Invalid dst parameter*" $error
} {1}

test distributed_isend-6.5 {Invalid tag parameter} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    catch {torch::distributed_isend $tensor 1 "invalid_tag"} error
    string match "*Invalid tag parameter*" $error
} {1}

test distributed_isend-6.6 {Unknown named parameter} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    catch {torch::distributed_isend -tensor $tensor -dst 1 -unknown_param 1} error
    string match "*Unknown parameter*" $error
} {1}

test distributed_isend-6.7 {Missing value for named parameter} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    catch {torch::distributed_isend -tensor $tensor -dst} error
    string match "*Missing value for parameter*" $error
} {1}

test distributed_isend-6.8 {Missing required named parameters} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    catch {torch::distributed_isend -tensor $tensor} error
    string match "*Required parameters missing*" $error
} {1}

test distributed_isend-6.9 {Invalid named dst parameter} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    catch {torch::distributed_isend -tensor $tensor -dst "invalid"} error
    string match "*Invalid -dst parameter*" $error
} {1}

test distributed_isend-6.10 {Invalid named tag parameter} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    catch {torch::distributed_isend -tensor $tensor -dst 1 -tag "invalid"} error
    string match "*Invalid -tag parameter*" $error
} {1}

# Test handle uniqueness
test distributed_isend-7.1 {Different dst produces different handles} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle1 [torch::distributed_isend $tensor 1]
    set handle2 [torch::distributed_isend $tensor 2]
    expr {$handle1 ne $handle2}
} {1}

test distributed_isend-7.2 {Different tags produce different handles} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle1 [torch::distributed_isend $tensor 1 10]
    set handle2 [torch::distributed_isend $tensor 1 20]
    expr {$handle1 ne $handle2}
} {1}

test distributed_isend-7.3 {Same parameters produce same handle format} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle1 [torch::distributed_isend $tensor 1 5]
    set handle2 [torch::distributed_isend $tensor 1 5]
    expr {$handle1 eq $handle2}
} {1}

# Test data type support
test distributed_isend-8.1 {Float tensor} {
    set tensor [torch::ones -shape {2 3} -dtype float32]
    set handle [torch::distributed_isend $tensor 1]
    regexp {isend_handle_dst1_tag0} $handle
} {1}

test distributed_isend-8.2 {Double tensor} {
    set tensor [torch::ones -shape {2 3} -dtype float64]
    set handle [torch::distributed_isend $tensor 1]
    regexp {isend_handle_dst1_tag0} $handle
} {1}

test distributed_isend-8.3 {Integer tensor} {
    set tensor [torch::ones -shape {2 3} -dtype int32]
    set handle [torch::distributed_isend $tensor 1]
    regexp {isend_handle_dst1_tag0} $handle
} {1}

# Test syntax equivalence
test distributed_isend-9.1 {Positional and named syntax equivalence} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle1 [torch::distributed_isend $tensor 1 5]
    set handle2 [torch::distributed_isend -tensor $tensor -dst 1 -tag 5]
    expr {$handle1 eq $handle2}
} {1}

test distributed_isend-9.2 {Named syntax parameter order independence} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle1 [torch::distributed_isend -tensor $tensor -dst 2 -tag 10]
    set handle2 [torch::distributed_isend -dst 2 -tag 10 -tensor $tensor]
    expr {$handle1 eq $handle2}
} {1}

test distributed_isend-9.3 {camelCase and snake_case equivalence} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle1 [torch::distributed_isend $tensor 1 5]
    set handle2 [torch::distributedIsend $tensor 1 5]
    expr {$handle1 eq $handle2}
} {1}

# Test edge cases
test distributed_isend-10.1 {Zero dst} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle [torch::distributed_isend $tensor 0]
    regexp {isend_handle_dst0_tag0} $handle
} {1}

test distributed_isend-10.2 {Large dst value} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle [torch::distributed_isend $tensor 1000]
    regexp {isend_handle_dst1000_tag0} $handle
} {1}

test distributed_isend-10.3 {Large tag value} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle [torch::distributed_isend $tensor 1 999999]
    regexp {isend_handle_dst1_tag999999} $handle
} {1}

test distributed_isend-10.4 {Negative tag value} {
    set tensor [create_test_tensor "test_tensor" {2 3}]
    set handle [torch::distributed_isend $tensor 1 -5]
    regexp {isend_handle_dst1_tag-5} $handle
} {1}

# Clean up
cleanupTests 