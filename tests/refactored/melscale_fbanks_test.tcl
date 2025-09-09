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

# Helper function to check if tensors are approximately equal
proc tensorsApproxEqual {tensor1 tensor2 tolerance} {
    set diff [torch::tensor_sub $tensor1 $tensor2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    return [expr {$max_val < $tolerance}]
}

# Helper function to check tensor shape
proc checkTensorShape {tensor expected_shape} {
    set shape [torch::tensor_shape $tensor]
    if {$shape != $expected_shape} {
        return 0
    }
    return 1
}

# Test cases for positional syntax
test melscale_fbanks-1.1 {Basic positional syntax} {
    set result [torch::melscale_fbanks 64 20 16000]
    checkTensorShape $result {20 64}
} {1}

test melscale_fbanks-1.2 {Positional syntax with f_min} {
    set result [torch::melscale_fbanks 64 20 16000 50.0]
    checkTensorShape $result {20 64}
} {1}

test melscale_fbanks-1.3 {Positional syntax with f_min and f_max} {
    set result [torch::melscale_fbanks 64 20 16000 50.0 7000.0]
    checkTensorShape $result {20 64}
} {1}

# Test cases for named parameter syntax
test melscale_fbanks-2.1 {Named parameter syntax - basic} {
    set result [torch::melscale_fbanks -nFreqs 64 -nMels 20 -sampleRate 16000]
    checkTensorShape $result {20 64}
} {1}

test melscale_fbanks-2.2 {Named parameter syntax with f_min} {
    set result [torch::melscale_fbanks -nFreqs 64 -nMels 20 -sampleRate 16000 -fMin 50.0]
    checkTensorShape $result {20 64}
} {1}

test melscale_fbanks-2.3 {Named parameter syntax with f_min and f_max} {
    set result [torch::melscale_fbanks -nFreqs 64 -nMels 20 -sampleRate 16000 -fMin 50.0 -fMax 7000.0]
    checkTensorShape $result {20 64}
} {1}

test melscale_fbanks-2.4 {Named parameter syntax with snake_case names} {
    set result [torch::melscale_fbanks -n_freqs 64 -n_mels 20 -sample_rate 16000]
    checkTensorShape $result {20 64}
} {1}

# Test cases for camelCase alias
test melscale_fbanks-3.1 {CamelCase alias - positional syntax} {
    set result [torch::melscaleFbanks 64 20 16000]
    checkTensorShape $result {20 64}
} {1}

test melscale_fbanks-3.2 {CamelCase alias - named parameter syntax} {
    set result [torch::melscaleFbanks -nFreqs 64 -nMels 20 -sampleRate 16000]
    checkTensorShape $result {20 64}
} {1}

# Consistency test - both syntaxes should produce the same result
test melscale_fbanks-4.1 {Consistency between positional and named syntax} {
    set result1 [torch::melscale_fbanks 64 20 16000 50.0 7000.0]
    set result2 [torch::melscale_fbanks -nFreqs 64 -nMels 20 -sampleRate 16000 -fMin 50.0 -fMax 7000.0]
    tensorsApproxEqual $result1 $result2 1e-6
} {1}

test melscale_fbanks-4.2 {Consistency between original and camelCase alias} {
    set result1 [torch::melscale_fbanks 64 20 16000]
    set result2 [torch::melscaleFbanks 64 20 16000]
    tensorsApproxEqual $result1 $result2 1e-6
} {1}

# Error handling tests
test melscale_fbanks-5.1 {Error: missing required parameters} {
    catch {torch::melscale_fbanks} err
    string match "*arguments*" $err
} {1}

test melscale_fbanks-5.2 {Error: invalid n_freqs} {
    catch {torch::melscale_fbanks -nFreqs -1 -nMels 20 -sampleRate 16000} err
    string match "*Invalid*" $err
} {1}

test melscale_fbanks-5.3 {Error: invalid n_mels} {
    catch {torch::melscale_fbanks -nFreqs 64 -nMels 0 -sampleRate 16000} err
    string match "*Invalid*" $err
} {1}

test melscale_fbanks-5.4 {Error: invalid sample_rate} {
    catch {torch::melscale_fbanks -nFreqs 64 -nMels 20 -sampleRate -1} err
    string match "*Invalid*" $err
} {1}

test melscale_fbanks-5.5 {Error: unknown parameter} {
    catch {torch::melscale_fbanks -nFreqs 64 -nMels 20 -sampleRate 16000 -unknown 123} err
    string match "*Unknown parameter*" $err
} {1}

cleanupTests
