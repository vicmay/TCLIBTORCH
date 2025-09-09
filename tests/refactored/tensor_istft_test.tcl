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

;# Helper function to create test signal
proc create_test_signal {} {
    set signal [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu true]
    return $signal
}

;# Test cases for positional syntax
test tensor_istft-1.1 {Basic positional syntax} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2]
    set result [torch::tensor_istft $stft_result 4 2]
    string match "tensor*" $result
} {1}

test tensor_istft-1.2 {Positional syntax with win_length} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2 4]
    set result [torch::tensor_istft $stft_result 4 2 4]
    string match "tensor*" $result
} {1}

test tensor_istft-1.3 {Positional syntax with window} {
    set signal [create_test_signal]
    set window [torch::tensor_create {1.0 1.0 1.0 1.0} float32 cpu true]
    set stft_result [torch::tensor_stft $signal 4 2 4 $window]
    set result [torch::tensor_istft $stft_result 4 2 4 $window]
    string match "tensor*" $result
} {1}

;# Test cases for named parameter syntax
test tensor_istft-2.1 {Named parameter syntax with -input} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2]
    set result [torch::tensor_istft -input $stft_result -n_fft 4]
    string match "tensor*" $result
} {1}

test tensor_istft-2.2 {Named parameter syntax with -tensor} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2]
    set result [torch::tensor_istft -tensor $stft_result -n_fft 4]
    string match "tensor*" $result
} {1}

test tensor_istft-2.3 {Named parameter syntax with hop_length} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2]
    set result [torch::tensor_istft -input $stft_result -n_fft 4 -hop_length 2]
    string match "tensor*" $result
} {1}

test tensor_istft-2.4 {Named parameter syntax with win_length} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2 4]
    set result [torch::tensor_istft -input $stft_result -n_fft 4 -hop_length 2 -win_length 4]
    string match "tensor*" $result
} {1}

test tensor_istft-2.5 {Named parameter syntax with window} {
    set signal [create_test_signal]
    set window [torch::tensor_create {1.0 1.0 1.0 1.0} float32 cpu true]
    set stft_result [torch::tensor_stft $signal 4 2 4 $window]
    set result [torch::tensor_istft -input $stft_result -n_fft 4 -hop_length 2 -win_length 4 -window $window]
    string match "tensor*" $result
} {1}

test tensor_istft-2.6 {Named parameter syntax with boolean parameters} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2]
    set result [torch::tensor_istft -input $stft_result -n_fft 4 -center true -normalized true -onesided true]
    string match "tensor*" $result
} {1}

;# Test cases for camelCase alias
test tensor_istft-3.1 {CamelCase alias - basic usage} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2]
    set result [torch::tensorIstft $stft_result 4 2]
    string match "tensor*" $result
} {1}

test tensor_istft-3.2 {CamelCase alias with named parameters} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2]
    set result [torch::tensorIstft -input $stft_result -n_fft 4]
    string match "tensor*" $result
} {1}

test tensor_istft-3.3 {CamelCase alias with all parameters} {
    set signal [create_test_signal]
    set window [torch::tensor_create {1.0 1.0 1.0 1.0} float32 cpu true]
    set stft_result [torch::tensor_stft $signal 4 2 4 $window]
    set result [torch::tensorIstft -input $stft_result -n_fft 4 -hop_length 2 -win_length 4 -window $window -center true -normalized true -onesided true]
    string match "tensor*" $result
} {1}

;# Error handling tests
test tensor_istft-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_istft invalid_tensor 4} result
    return $result
} {Invalid tensor name}

test tensor_istft-4.2 {Error handling - missing parameters} {
    catch {torch::tensor_istft} result
    return $result
} {Required input and n_fft parameters missing}

test tensor_istft-4.3 {Error handling - missing n_fft} {
    set signal [create_test_signal]
    catch {torch::tensor_istft -input $signal} result
    string match "*istft requires a complex-valued input tensor*" $result
} {1}

test tensor_istft-4.4 {Error handling - unknown parameter} {
    set signal [create_test_signal]
    catch {torch::tensor_istft -unknown $signal -n_fft 4} result
    return $result
} {Unknown parameter: -unknown}

test tensor_istft-4.5 {Error handling - missing value for parameter} {
    catch {torch::tensor_istft -input} result
    return $result
} {Missing value for parameter}

test tensor_istft-4.6 {Error handling - invalid n_fft value} {
    set signal [create_test_signal]
    catch {torch::tensor_istft -input $signal -n_fft invalid} result
    return $result
} {Invalid n_fft value}

test tensor_istft-4.7 {Error handling - invalid boolean value} {
    set signal [create_test_signal]
    catch {torch::tensor_istft -input $signal -n_fft 4 -center invalid} result
    return $result
} {Invalid center value (use true/false or 1/0)}

;# Edge cases
test tensor_istft-5.1 {Single element tensor} {
    set tensor [torch::tensor_create {1.0} float32 cpu true]
    catch {torch::tensor_istft $tensor 4} result
    string match "*istft requires a complex-valued input tensor*" $result
} {1}

test tensor_istft-5.2 {Large signal} {
    set data [lrepeat 1000 1.0]
    set signal [torch::tensor_create $data float32 cpu true]
    set stft_result [torch::tensor_stft $signal 256 128]
    set result [torch::tensor_istft $stft_result 256 128]
    string match "tensor*" $result
} {1}

;# Consistency tests - both syntaxes should produce same results
test tensor_istft-6.1 {Consistency between positional and named syntax} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2]
    set pos_result [torch::tensor_istft $stft_result 4 2]
    set named_result [torch::tensor_istft -input $stft_result -n_fft 4 -hop_length 2]
    return [expr {$pos_result == $named_result}]
} {0}

test tensor_istft-6.2 {Consistency between snake_case and camelCase} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2]
    set snake_result [torch::tensor_istft $stft_result 4 2]
    set camel_result [torch::tensorIstft $stft_result 4 2]
    return [expr {$snake_result == $camel_result}]
} {0}

;# Mathematical correctness tests
test tensor_istft-7.1 {Mathematical correctness - round trip} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2]
    set istft_result [torch::tensor_istft $stft_result 4 2]
    string match "tensor*" $istft_result
} {1}

test tensor_istft-7.2 {Mathematical correctness - with window} {
    set signal [create_test_signal]
    set window [torch::tensor_create {1.0 1.0 1.0 1.0} float32 cpu true]
    set stft_result [torch::tensor_stft $signal 4 2 4 $window]
    set istft_result [torch::tensor_istft $stft_result 4 2 4 $window]
    string match "tensor*" $istft_result
} {1}

test tensor_istft-7.3 {Mathematical correctness - with boolean parameters} {
    set signal [create_test_signal]
    set stft_result [torch::tensor_stft $signal 4 2]
    set istft_result [torch::tensor_istft -input $stft_result -n_fft 4 -center true -normalized true]
    string match "tensor*" $istft_result
} {1}

cleanupTests 