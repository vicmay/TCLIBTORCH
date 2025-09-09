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

# Helper function to create test signal
proc create_test_signal {} {
    # Create a simple sine wave
    set t [torch::tensor_create {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99} float32]
    set freq 440.0  ;# 440 Hz
    set sample_rate 16000.0
    set scale [expr {2.0 * 3.14159 * $freq / $sample_rate}]
    set scale_tensor [torch::tensor_create $scale float32]
    set t [torch::tensor_mul $t $scale_tensor]
    set signal [torch::sin $t]
    return $signal
}

# Test cases for positional syntax
test spectrogram-1.1 {Basic positional syntax} {
    set signal [create_test_signal]
    set result [torch::spectrogram $signal]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] > 0 && [lindex $shape 1] > 0}
} {1}

test spectrogram-1.2 {Positional syntax with n_fft} {
    set signal [create_test_signal]
    set result [torch::spectrogram $signal 32]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 17}  ;# (n_fft/2 + 1)
} {1}

test spectrogram-1.3 {Positional syntax with n_fft and hop_length} {
    set signal [create_test_signal]
    set result [torch::spectrogram $signal 32 16]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 1] > 0}
} {1}

test spectrogram-1.4 {Positional syntax with all parameters} {
    set signal [create_test_signal]
    set window [torch::hann_window 32]
    set result [torch::spectrogram $signal 32 16 32 $window]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 17 && [lindex $shape 1] > 0}
} {1}

# Test cases for named syntax
test spectrogram-2.1 {Named syntax with input only} {
    set signal [create_test_signal]
    set result [torch::spectrogram -input $signal]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] > 0 && [lindex $shape 1] > 0}
} {1}

test spectrogram-2.2 {Named syntax with nFft} {
    set signal [create_test_signal]
    set result [torch::spectrogram -input $signal -nFft 32]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 17}  ;# (n_fft/2 + 1)
} {1}

test spectrogram-2.3 {Named syntax with hopLength} {
    set signal [create_test_signal]
    set result [torch::spectrogram -input $signal -nFft 32 -hopLength 16]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 1] > 0}
} {1}

test spectrogram-2.4 {Named syntax with all parameters} {
    set signal [create_test_signal]
    set window [torch::hann_window 32]
    set result [torch::spectrogram -input $signal -nFft 32 -hopLength 16 -winLength 32 -window $window]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 17 && [lindex $shape 1] > 0}
} {1}

# Test cases for camelCase alias
test spectrogram-3.1 {CamelCase alias} {
    set signal [create_test_signal]
    set result [torch::spectrogram -input $signal -nFft 32 -hopLength 16 -winLength 32]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 17 && [lindex $shape 1] > 0}
} {1}

# Error handling tests
test spectrogram-4.1 {Error on missing input} {
    catch {torch::spectrogram} msg
    set msg
} {Error in spectrogram: Wrong number of arguments}

test spectrogram-4.2 {Error on invalid tensor} {
    catch {torch::spectrogram "not_a_tensor"} msg
    set msg
} {Error in spectrogram: Invalid tensor}

test spectrogram-4.3 {Error on negative hop_length} {
    set signal [create_test_signal]
    catch {torch::spectrogram $signal 32 -1} msg
    set msg
} {Error in spectrogram: hop_length must be positive}

test spectrogram-4.4 {Error on invalid window tensor} {
    set signal [create_test_signal]
    catch {torch::spectrogram $signal 32 16 32 "not_a_tensor"} msg
    set msg
} {Error in spectrogram: Invalid tensor}

cleanupTests 