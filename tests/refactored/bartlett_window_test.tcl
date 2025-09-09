#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

if {[catch {load ../../build/libtorchtcl.so} err]} {
    puts "Failed to load libtorchtcl.so: $err"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax

test bartlett-window-1.1 {positional create basic} {
    set window [torch::bartlett_window 5]
    string match "tensor*" $window
} {1}

test bartlett-window-1.2 {positional with dtype} {
    set window [torch::bartlett_window 4 float64]
    string match "tensor*" $window
} {1}

test bartlett-window-1.3 {positional with dtype and device} {
    set window [torch::bartlett_window 6 float32 cpu]
    string match "tensor*" $window
} {1}

test bartlett-window-1.4 {positional with all parameters} {
    set window [torch::bartlett_window 4 float32 cpu 1]
    string match "tensor*" $window
} {1}

test bartlett-window-1.5 {positional non-periodic} {
    set window [torch::bartlett_window 4 float32 cpu 0]
    string match "tensor*" $window
} {1}

# Test 2: Named parameter syntax

test bartlett-window-2.1 {named create basic} {
    set window [torch::bartlett_window -window_length 5]
    string match "tensor*" $window
} {1}

test bartlett-window-2.2 {named with windowLength} {
    set window [torch::bartlett_window -windowLength 4]
    string match "tensor*" $window
} {1}

test bartlett-window-2.3 {named with length alias} {
    set window [torch::bartlett_window -length 3]
    string match "tensor*" $window
} {1}

test bartlett-window-2.4 {named with dtype} {
    set window [torch::bartlett_window -window_length 4 -dtype float64]
    string match "tensor*" $window
} {1}

test bartlett-window-2.5 {named with device} {
    set window [torch::bartlett_window -window_length 5 -device cpu]
    string match "tensor*" $window
} {1}

test bartlett-window-2.6 {named with periodic} {
    set window [torch::bartlett_window -window_length 4 -periodic 0]
    string match "tensor*" $window
} {1}

test bartlett-window-2.7 {named with all parameters} {
    set window [torch::bartlett_window -window_length 6 -dtype float32 -device cpu -periodic 1]
    string match "tensor*" $window
} {1}

# Test 3: camelCase alias

test bartlett-window-3.1 {camelCase alias basic} {
    set window [torch::bartlettWindow -windowLength 4]
    string match "tensor*" $window
} {1}

test bartlett-window-3.2 {camelCase with snake_case parameters} {
    set window [torch::bartlettWindow -window_length 5]
    string match "tensor*" $window
} {1}

test bartlett-window-3.3 {camelCase with mixed parameters} {
    set window [torch::bartlettWindow -windowLength 4 -dtype float64 -periodic 0]
    string match "tensor*" $window
} {1}

# Test 4: Basic correctness

test bartlett-window-4.1 {basic shape correctness} {
    set window [torch::bartlett_window 5]
    set shape [torch::tensor_shape $window]
    expr {$shape eq "5"}
} {1}

test bartlett-window-4.2 {basic dtype correctness} {
    set window [torch::bartlett_window -window_length 4 -dtype float64]
    set dtype [torch::tensor_dtype $window]
    expr {$dtype eq "Float64"}
} {1}

# Test 5: Different window lengths

test bartlett-window-5.1 {window length 1} {
    set window [torch::bartlett_window 1]
    set shape [torch::tensor_shape $window]
    expr {$shape eq "1"}
} {1}

test bartlett-window-5.2 {window length 10} {
    set window [torch::bartlett_window 10]
    set shape [torch::tensor_shape $window]
    expr {$shape eq "10"}
} {1}

# Test 6: Consistency between syntaxes

test bartlett-window-6.1 {consistency - positional vs named} {
    set window1 [torch::bartlett_window 4 float32 cpu 1]
    set window2 [torch::bartlett_window -window_length 4 -dtype float32 -device cpu -periodic 1]
    set shape1 [torch::tensor_shape $window1]
    set shape2 [torch::tensor_shape $window2]
    expr {$shape1 eq $shape2}
} {1}

test bartlett-window-6.2 {consistency - snake_case vs camelCase} {
    set window1 [torch::bartlett_window -window_length 5 -periodic 0]
    set window2 [torch::bartlettWindow -windowLength 5 -periodic 0]
    set shape1 [torch::tensor_shape $window1]
    set shape2 [torch::tensor_shape $window2]
    expr {$shape1 eq $shape2}
} {1}

# Test 7: Error handling

test bartlett-window-7.1 {missing window_length} -body {
    torch::bartlett_window
} -returnCodes error -match glob -result *pairs*

test bartlett-window-7.2 {invalid window_length - zero} -body {
    torch::bartlett_window 0
} -returnCodes error -match glob -result *positive*

test bartlett-window-7.3 {invalid window_length - string} -body {
    torch::bartlett_window invalid
} -returnCodes error -match glob -result *integer*

test bartlett-window-7.4 {missing value for named parameter} -body {
    torch::bartlett_window -window_length
} -returnCodes error -match glob -result *pairs*

test bartlett-window-7.5 {unknown parameter} -body {
    torch::bartlett_window -window_length 4 -unknown value
} -returnCodes error -result "Unknown parameter: -unknown"

test bartlett-window-7.6 {invalid dtype} -body {
    torch::bartlett_window -window_length 4 -dtype invalid_dtype
} -returnCodes error -match glob -result *dtype*

test bartlett-window-7.7 {invalid periodic value} -body {
    torch::bartlett_window -window_length 4 -periodic invalid
} -returnCodes error -match glob -result *periodic*

# Test 8: Data type support

test bartlett-window-8.1 {float32 dtype} {
    set window [torch::bartlett_window -window_length 3 -dtype float32]
    string match "tensor*" $window
} {1}

test bartlett-window-8.2 {float64 dtype} {
    set window [torch::bartlett_window -window_length 3 -dtype float64]
    string match "tensor*" $window
} {1}

test bartlett-window-8.3 {double alias} {
    set window [torch::bartlett_window -window_length 3 -dtype double]
    string match "tensor*" $window
} {1}

test bartlett-window-8.4 {float alias} {
    set window [torch::bartlett_window -window_length 3 -dtype float]
    string match "tensor*" $window
} {1}

# Test 9: Edge cases

test bartlett-window-9.1 {minimum window length} {
    set window [torch::bartlett_window 1]
    set shape [torch::tensor_shape $window]
    expr {$shape eq "1"}
} {1}

test bartlett-window-9.2 {even window length} {
    set window [torch::bartlett_window 4]
    set shape [torch::tensor_shape $window]
    expr {$shape eq "4"}
} {1}

test bartlett-window-9.3 {odd window length} {
    set window [torch::bartlett_window 5]
    set shape [torch::tensor_shape $window]
    expr {$shape eq "5"}
} {1}

# Test 10: Integration with other commands

test bartlett-window-10.1 {integration with tensor_shape} {
    set window [torch::bartlett_window -window_length 6]
    set shape [torch::tensor_shape $window]
    expr {$shape eq "6"}
} {1}

test bartlett-window-10.2 {integration with tensor_dtype} {
    set window [torch::bartlett_window -window_length 4 -dtype float64]
    set dtype [torch::tensor_dtype $window]
    expr {$dtype eq "Float64"}
} {1}

cleanupTests 