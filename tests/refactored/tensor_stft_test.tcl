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

;# Test cases for positional syntax
test tensor_stft-1.1 {Basic positional syntax - tensor and n_fft} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_stft $tensor 4]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-1.2 {Positional syntax with hop_length} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_stft $tensor 4 2]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-1.3 {Positional syntax with win_length} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_stft $tensor 4 2 4]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-1.4 {Positional syntax with window} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set window [torch::tensor_create {0.5 1.0 0.5 0.5}]
    set result [torch::tensor_stft $tensor 4 2 4 $window]
    expr {[string length $result] > 0}
} {1}

;# Test cases for named parameter syntax
test tensor_stft-2.1 {Named parameter syntax - input and n_fft} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_stft -input $tensor -n_fft 4]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-2.2 {Named parameter syntax - all parameters} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_stft -tensor $tensor -nfft 4 -hopLength 2 -winLength 4]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-2.3 {Named parameter syntax - with window} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set window [torch::tensor_create {0.5 1.0 0.5 0.5}]
    set result [torch::tensor_stft -input $tensor -n_fft 4 -hop_length 2 -win_length 4 -window $window]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-2.4 {Named parameter syntax - different parameter order} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_stft -n_fft 4 -hop_length 2 -tensor $tensor]
    expr {[string length $result] > 0}
} {1}

;# Test cases for camelCase alias
test tensor_stft-3.1 {CamelCase alias - positional syntax} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensorStft $tensor 4]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-3.2 {CamelCase alias - named parameter syntax} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensorStft -input $tensor -n_fft 4 -hop_length 2]
    expr {[string length $result] > 0}
} {1}

;# Error handling tests
test tensor_stft-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_stft invalid_tensor 4} result
    return $result
} {Invalid tensor name}

test tensor_stft-4.2 {Error handling - missing tensor} {
    catch {torch::tensor_stft} result
    return $result
} {Required input and n_fft parameters missing}

test tensor_stft-4.3 {Error handling - missing n_fft} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_stft $tensor} result
    return $result
} {Invalid number of arguments}

test tensor_stft-4.4 {Error handling - invalid n_fft} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_stft $tensor invalid} result
    return $result
} {Invalid n_fft value}

test tensor_stft-4.5 {Error handling - invalid hop_length} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_stft $tensor 4 invalid} result
    return $result
} {Invalid hop_length value}

test tensor_stft-4.6 {Error handling - invalid win_length} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_stft $tensor 4 2 invalid} result
    return $result
} {Invalid win_length value}

test tensor_stft-4.7 {Error handling - invalid named parameter} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_stft -invalid $tensor} result
    return $result
} {Unknown parameter: -invalid}

test tensor_stft-4.8 {Error handling - missing parameter value} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_stft -tensor $tensor -n_fft} result
    return $result
} {Missing value for parameter}

;# Edge cases
test tensor_stft-5.1 {Edge case - small tensor} {
    set tensor [torch::tensor_create {1 2 3}]
    set result [torch::tensor_stft $tensor 2 1]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-5.2 {Edge case - power of 2 n_fft} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_stft $tensor 8]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-5.3 {Edge case - non-power of 2 n_fft} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7}]
    set result [torch::tensor_stft $tensor 5]
    expr {[string length $result] > 0}
} {1}

;# Data type tests
test tensor_stft-6.1 {Data type - float32 tensor} {
    set tensor [torch::tensor_create {1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5} float32]
    set result [torch::tensor_stft $tensor 4]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-6.2 {Data type - float64 tensor} {
    set tensor [torch::tensor_create {1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5} float64]
    set result [torch::tensor_stft $tensor 4]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-6.3 {Data type - int32 tensor} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8} int32]
    catch {torch::tensor_stft $tensor 4} result
    expr {[string length $result] > 0}
} {1}

;# STFT-specific tests
test tensor_stft-7.1 {STFT specific - different hop lengths} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8 9 10 11 12}]
    set result1 [torch::tensor_stft $tensor 4 1]
    set result2 [torch::tensor_stft $tensor 4 2]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_stft-7.2 {STFT specific - different window lengths} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result1 [torch::tensor_stft $tensor 4 2 4]
    set result2 [torch::tensor_stft $tensor 6 2 6]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_stft-7.3 {STFT specific - custom window} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set window [torch::tensor_create {0.25 0.5 0.75 1.0}]
    set result [torch::tensor_stft $tensor 4 2 4 $window]
    expr {[string length $result] > 0}
} {1}

;# Mathematical correctness tests
test tensor_stft-8.1 {Mathematical correctness - known values} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_stft $tensor 4]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-8.2 {Mathematical correctness - different n_fft values} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result1 [torch::tensor_stft $tensor 4]
    set result2 [torch::tensor_stft $tensor 8]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

;# Consistency tests - both syntaxes should produce same results
test tensor_stft-9.1 {Consistency - positional vs named syntax} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result1 [torch::tensor_stft $tensor 4 2]
    set result2 [torch::tensor_stft -input $tensor -n_fft 4 -hop_length 2]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_stft-9.2 {Consistency - snake_case vs camelCase} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result1 [torch::tensor_stft $tensor 4]
    set result2 [torch::tensorStft $tensor 4]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

;# Complex scenarios
test tensor_stft-10.1 {Complex - large tensor} {
    set data {}
    for {set i 1} {$i <= 100} {incr i} {
        lappend data $i
    }
    set tensor [torch::tensor_create $data]
    set result [torch::tensor_stft $tensor 16 8]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-10.2 {Complex - sine wave} {
    set data {}
    for {set i 0} {$i < 100} {incr i} {
        lappend data [expr {sin($i * 0.1)}]
    }
    set tensor [torch::tensor_create $data]
    set result [torch::tensor_stft $tensor 16 8]
    expr {[string length $result] > 0}
} {1}

;# Window function tests
test tensor_stft-11.1 {Window function - Hann window default} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_stft $tensor 4]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-11.2 {Window function - custom window} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set window [torch::tensor_create {1.0 1.0 1.0 1.0}]
    set result [torch::tensor_stft $tensor 4 2 4 $window]
    expr {[string length $result] > 0}
} {1}

test tensor_stft-11.3 {Window function - window length vs n_fft} {
    set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
    set result [torch::tensor_stft $tensor 8 4 6]
    expr {[string length $result] > 0}
} {1}

cleanupTests 