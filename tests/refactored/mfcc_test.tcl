#!/usr/bin/env tclsh

# Test file for torch::mfcc command with dual syntax support
# Tests both positional and named parameter syntax

package require tcltest
namespace import tcltest::*

# Load the LibTorch TCL extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test suite for torch::mfcc
test mfcc-1.1 {Basic positional syntax} {
    # Create a mel spectrogram tensor (2D)
    set data [list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2]
    set mel_spec [torch::tensor_create -data $data -dtype float32 -device cpu]
    set mel_spec [torch::tensor_reshape $mel_spec {3 4}]
    
    # Test basic mfcc with default parameters
    set result [torch::mfcc $mel_spec]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test mfcc-1.2 {Positional syntax with n_mfcc} {
    # Create a mel spectrogram tensor (2D)
    set data [list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2]
    set mel_spec [torch::tensor_create -data $data -dtype float32 -device cpu]
    set mel_spec [torch::tensor_reshape $mel_spec {3 4}]
    
    # Test mfcc with custom n_mfcc
    set result [torch::mfcc $mel_spec 8]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test mfcc-1.3 {Positional syntax with n_mfcc and dct_type} {
    # Create a mel spectrogram tensor (2D)
    set data [list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2]
    set mel_spec [torch::tensor_create -data $data -dtype float32 -device cpu]
    set mel_spec [torch::tensor_reshape $mel_spec {3 4}]
    
    # Test mfcc with custom n_mfcc and dct_type
    set result [torch::mfcc $mel_spec 8 2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test mfcc-2.1 {Named parameter syntax - basic} {
    # Create a mel spectrogram tensor (2D)
    set data [list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2]
    set mel_spec [torch::tensor_create -data $data -dtype float32 -device cpu]
    set mel_spec [torch::tensor_reshape $mel_spec {3 4}]
    
    # Test basic named parameter syntax
    set result [torch::mfcc -spectrogram $mel_spec]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test mfcc-2.2 {Named parameter syntax - with n_mfcc} {
    # Create a mel spectrogram tensor (2D)
    set data [list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2]
    set mel_spec [torch::tensor_create -data $data -dtype float32 -device cpu]
    set mel_spec [torch::tensor_reshape $mel_spec {3 4}]
    
    # Test named parameter syntax with n_mfcc
    set result [torch::mfcc -spectrogram $mel_spec -nMfcc 8]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test mfcc-2.3 {Named parameter syntax - with snake_case parameters} {
    # Create a mel spectrogram tensor (2D)
    set data [list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2]
    set mel_spec [torch::tensor_create -data $data -dtype float32 -device cpu]
    set mel_spec [torch::tensor_reshape $mel_spec {3 4}]
    
    # Test named parameter syntax with snake_case parameters
    set result [torch::mfcc -spectrogram $mel_spec -n_mfcc 8 -dct_type 2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test mfcc-2.4 {Named parameter syntax - with camelCase parameters} {
    # Create a mel spectrogram tensor (2D)
    set data [list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2]
    set mel_spec [torch::tensor_create -data $data -dtype float32 -device cpu]
    set mel_spec [torch::tensor_reshape $mel_spec {3 4}]
    
    # Test named parameter syntax with camelCase parameters
    set result [torch::mfcc -spectrogram $mel_spec -nMfcc 8 -dctType 2]
    
    # Verify result is a valid tensor handle
    expr {[string length $result] > 0}
} {1}

test mfcc-4.1 {Error handling - invalid spectrogram} {
    # Test with invalid tensor
    catch {torch::mfcc "invalid_tensor"} err
    
    # Verify error message
    string match "*Invalid spectrogram tensor*" $err
} {1}

test mfcc-4.2 {Error handling - invalid n_mfcc} {
    # Create a mel spectrogram tensor (2D)
    set data [list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2]
    set mel_spec [torch::tensor_create -data $data -dtype float32 -device cpu]
    set mel_spec [torch::tensor_reshape $mel_spec {3 4}]
    
    # Test with invalid n_mfcc
    catch {torch::mfcc $mel_spec "not_a_number"} err
    
    # Verify error message
    string match "*Invalid n_mfcc value*" $err
} {1}

test mfcc-4.3 {Error handling - invalid parameter} {
    # Create a mel spectrogram tensor (2D)
    set data [list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2]
    set mel_spec [torch::tensor_create -data $data -dtype float32 -device cpu]
    set mel_spec [torch::tensor_reshape $mel_spec {3 4}]
    
    # Test with invalid parameter
    catch {torch::mfcc -invalid $mel_spec} err
    
    # Verify error message
    string match "*Unknown parameter*" $err
} {1}

cleanupTests
