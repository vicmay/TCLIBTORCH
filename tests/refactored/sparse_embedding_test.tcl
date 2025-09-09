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

# Helper function to check tensor shape
proc check_tensor_shape {tensor expected_shape} {
    set shape [torch::tensor_shape $tensor]
    if {$shape != $expected_shape} {
        error "Expected shape $expected_shape but got $shape"
    }
}

# Test cases for positional syntax
test sparse_embedding-1.1 {Basic positional syntax} {
    set input [torch::tensor_create {0 1 2} int64 cpu]
    set result [torch::sparse_embedding $input 10 5 -1]
    check_tensor_shape $result {3 5}
    expr {1}
} {1}

test sparse_embedding-1.2 {Positional syntax with padding_idx} {
    set input [torch::tensor_create {0 1 2} int64 cpu]
    set result [torch::sparse_embedding $input 10 5 0]
    check_tensor_shape $result {3 5}
    # Check if padding index is zeroed
    set first_row_str [torch::tensor_print [torch::tensor_reshape -input $result -shape {3 5}]]
    expr {[string first "0" $first_row_str] >= 0}
} {1}

# Test cases for named parameter syntax
test sparse_embedding-2.1 {Named parameter syntax} {
    set input [torch::tensor_create {0 1 2} int64 cpu]
    set result [torch::sparse_embedding -input $input -num_embeddings 10 -embedding_dim 5 -padding_idx -1]
    check_tensor_shape $result {3 5}
    expr {1}
} {1}

test sparse_embedding-2.2 {Named parameter syntax with padding_idx} {
    set input [torch::tensor_create {0 1 2} int64 cpu]
    set result [torch::sparse_embedding -input $input -num_embeddings 10 -embedding_dim 5 -padding_idx 0]
    check_tensor_shape $result {3 5}
    # Check if padding index is zeroed
    set first_row_str [torch::tensor_print [torch::tensor_reshape -input $result -shape {3 5}]]
    expr {[string first "0" $first_row_str] >= 0}
} {1}

# Test cases for camelCase alias
test sparse_embedding-3.1 {CamelCase alias} {
    set input [torch::tensor_create {0 1 2} int64 cpu]
    set result [torch::sparseEmbedding -input $input -num_embeddings 10 -embedding_dim 5 -padding_idx -1]
    check_tensor_shape $result {3 5}
    expr {1}
} {1}

# Error handling tests
test sparse_embedding-4.1 {Error: Invalid input tensor} {
    catch {torch::sparse_embedding invalid_tensor 10 5 -1} err
    set err
} {Invalid input tensor}

test sparse_embedding-4.2 {Error: Invalid num_embeddings} {
    set input [torch::tensor_create {0 1 2} int64 cpu]
    catch {torch::sparse_embedding -input $input -num_embeddings -1 -embedding_dim 5 -padding_idx -1} err
    set err
} {Required parameters missing: input, num_embeddings, embedding_dim}

test sparse_embedding-4.3 {Error: Invalid embedding_dim} {
    set input [torch::tensor_create {0 1 2} int64 cpu]
    catch {torch::sparse_embedding -input $input -num_embeddings 10 -embedding_dim 0 -padding_idx -1} err
    set err
} {Required parameters missing: input, num_embeddings, embedding_dim}

test sparse_embedding-4.4 {Error: Missing required parameters} {
    set input [torch::tensor_create {0 1 2} int64 cpu]
    catch {torch::sparse_embedding -input $input} err
    set err
} {Required parameters missing: input, num_embeddings, embedding_dim}

test sparse_embedding-4.5 {Error: Unknown parameter} {
    set input [torch::tensor_create {0 1 2} int64 cpu]
    catch {torch::sparse_embedding -input $input -num_embeddings 10 -embedding_dim 5 -invalid_param 0} err
    set err
} {Unknown parameter: -invalid_param}

cleanupTests