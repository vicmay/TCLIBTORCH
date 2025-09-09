#!/usr/bin/env tclsh

# Load the LibTorch TCL extension
if {[catch {load ./libtorchtcl.so} err]} {
    puts "Error loading libtorchtcl.so: $err"
    exit 1
}

puts "Testing LibTorch TCL Extension - Batch 5: Advanced Neural Network Layers"
puts "============================================================================"

# Test Transformer Components
puts "\n=== Testing Transformer Components ==="

# Test Multi-Head Attention
puts "Testing multihead_attention..."
if {[catch {
    set query [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set query [torch::tensor_reshape $query {2 2 2}]
    set key [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set key [torch::tensor_reshape $key {2 2 2}]
    set value [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set value [torch::tensor_reshape $value {2 2 2}]
    set mha_result [torch::multihead_attention $query $key $value 2 1]
    puts "MHA output shape: [torch::tensor_shape $mha_result]"
} err]} {
    puts "Error in multihead_attention: $err"
}

# Test Positional Encoding
puts "Testing positional_encoding..."
if {[catch {
    set pos_result [torch::positional_encoding 4 8 0.1]
    puts "Positional encoding shape: [torch::tensor_shape $pos_result]"
} err]} {
    puts "Error in positional_encoding: $err"
}

# Test Transformer Encoder Layer
puts "Testing transformer_encoder_layer..."
if {[catch {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} float32]
    set src [torch::tensor_reshape $src {2 2 4}]
    set enc_result [torch::transformer_encoder_layer $src 4 2 4 0.1]
    puts "Encoder layer output shape: [torch::tensor_shape $enc_result]"
} err]} {
    puts "Error in transformer_encoder_layer: $err"
}

# Test Transformer Decoder Layer
puts "Testing transformer_decoder_layer..."
if {[catch {
    set tgt [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} float32]
    set tgt [torch::tensor_reshape $tgt {2 2 4}]
    set memory [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} float32]
    set memory [torch::tensor_reshape $memory {2 2 4}]
    set dec_result [torch::transformer_decoder_layer $tgt $memory 4 2 4 0.1]
    puts "Decoder layer output shape: [torch::tensor_shape $dec_result]"
} err]} {
    puts "Error in transformer_decoder_layer: $err"
}

# Test Transformer Encoder
puts "Testing transformer_encoder..."
if {[catch {
    set src [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} float32]
    set src [torch::tensor_reshape $src {2 2 4}]
    set trans_enc_result [torch::transformer_encoder $src 4 2 3 4]
    puts "Transformer encoder output shape: [torch::tensor_shape $trans_enc_result]"
} err]} {
    puts "Error in transformer_encoder: $err"
}

# Test Transformer Decoder
puts "Testing transformer_decoder..."
if {[catch {
    set tgt [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} float32]
    set tgt [torch::tensor_reshape $tgt {2 2 4}]
    set memory [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} float32]
    set memory [torch::tensor_reshape $memory {2 2 4}]
    set trans_dec_result [torch::transformer_decoder $tgt $memory 4 2 3 4]
    puts "Transformer decoder output shape: [torch::tensor_shape $trans_dec_result]"
} err]} {
    puts "Error in transformer_decoder: $err"
}

# Test Scaled Dot Product Attention
puts "Testing scaled_dot_product_attention..."
if {[catch {
    set query2 [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set query2 [torch::tensor_reshape $query2 {1 1 4}]
    set key2 [torch::tensor_create {5.0 6.0 7.0 8.0} float32]
    set key2 [torch::tensor_reshape $key2 {1 1 4}]
    set value2 [torch::tensor_create {9.0 10.0 11.0 12.0} float32]
    set value2 [torch::tensor_reshape $value2 {1 1 4}]
    set attn_result [torch::scaled_dot_product_attention $query2 $key2 $value2]
    puts "Scaled dot product attention output shape: [torch::tensor_shape $attn_result]"
} err]} {
    puts "Error in scaled_dot_product_attention: $err"
}

puts "\n=== Testing Embedding Layers ==="

# Test Embedding
puts "Testing embedding..."
if {[catch {
    set indices [torch::tensor_create {0 1 2 3} int64]
    set emb_result [torch::embedding $indices 10 4 -1]
    puts "Embedding output shape: [torch::tensor_shape $emb_result]"
} err]} {
    puts "Error in embedding: $err"
}

# Test Embedding Bag
puts "Testing embedding_bag..."
if {[catch {
    set indices [torch::tensor_create {0 1 2} int64]
    set weight [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0} float32]
    set weight [torch::tensor_reshape $weight {3 4}]
    set offsets [torch::tensor_create {0 2} int64]
    set per_sample_weights [torch::tensor_create {1.0 1.0 1.0} float32]
    set embag_result [torch::embedding_bag $indices $weight $offsets 0 $per_sample_weights]
    puts "Embedding bag output shape: [torch::tensor_shape $embag_result]"
} err]} {
    puts "Error in embedding_bag: $err"
}

# Test Sparse Embedding
puts "Testing sparse_embedding..."
if {[catch {
    set indices [torch::tensor_create {0 1 2 3} int64]
    set sparse_emb_result [torch::sparse_embedding $indices 10 4 -1]
    puts "Sparse embedding output shape: [torch::tensor_shape $sparse_emb_result]"
} err]} {
    puts "Error in sparse_embedding: $err"
}

puts "\n==========================================="
puts "Batch 5 Advanced Neural Network Layers test completed!"
puts "Total commands tested: 10"
puts "- 7 Transformer Components"
puts "- 3 Embedding Layers"
puts "===========================================" 