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

# Test 1: Basic positional syntax
test fold-1.1 {Basic positional syntax - simple case} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set result [torch::fold $input {2 2} {2 2}]
    expr {[string length $result] > 0}
} {1}

test fold-1.2 {Positional syntax with optional parameters} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set result [torch::fold $input {2 2} {2 2} {1 1} {0 0} {1 1}]
    expr {[string length $result] > 0}
} {1}

test fold-1.3 {Positional syntax with larger kernel} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0} -shape {1 9 1} -dtype float32]
    set result [torch::fold $input {3 3} {3 3}]
    expr {[string length $result] > 0}
} {1}

# Test 2: Named parameter syntax
test fold-2.1 {Named parameter syntax - basic} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set result [torch::fold -input $input -output_size {2 2} -kernel_size {2 2}]
    expr {[string length $result] > 0}
} {1}

test fold-2.2 {Named parameter syntax with camelCase} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set result [torch::fold -input $input -outputSize {2 2} -kernelSize {2 2}]
    expr {[string length $result] > 0}
} {1}

test fold-2.3 {Named parameter syntax with all parameters} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set result [torch::fold -input $input -output_size {2 2} -kernel_size {2 2} -dilation {1 1} -padding {0 0} -stride {1 1}]
    expr {[string length $result] > 0}
} {1}

test fold-2.4 {Named parameter syntax with tensor parameter} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set result [torch::fold -tensor $input -output_size {2 2} -kernel_size {2 2}]
    expr {[string length $result] > 0}
} {1}

# Test 3: CamelCase alias
test fold-3.1 {CamelCase alias basic functionality} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set result [torch::Fold $input {2 2} {2 2}]
    expr {[string length $result] > 0}
} {1}

test fold-3.2 {CamelCase alias with named parameters} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set result [torch::Fold -input $input -output_size {2 2} -kernel_size {2 2}]
    expr {[string length $result] > 0}
} {1}

test fold-3.3 {CamelCase alias with camelCase parameters} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set result [torch::Fold -input $input -outputSize {2 2} -kernelSize {2 2}]
    expr {[string length $result] > 0}
} {1}

# Test 4: Command existence verification
test fold-4.1 {Verify torch::fold command exists} {
    info commands torch::fold
} {::torch::fold}

test fold-4.2 {Verify torch::Fold camelCase alias exists} {
    info commands torch::Fold
} {::torch::Fold}

# Test 5: Different configurations
test fold-5.1 {Different output size} {
    # Create a tensor with shape [1, 8, 3] which can be folded into [1, 2, 4] with kernel_size [2, 2]
    # This matches the expected calculation: input.size(2) = 3 sliding blocks
    set data {}
    for {set i 1} {$i <= 24} {incr i} {
        lappend data $i.0
    }
    set input [torch::tensorCreate -data $data -shape {1 8 3} -dtype float32]
    set result [torch::fold $input {2 4} {2 2}]
    expr {[string length $result] > 0}
} {1}

test fold-5.2 {Rectangular kernel} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {1 6 1} -dtype float32]
    set result [torch::fold $input {3 2} {3 2}]
    expr {[string length $result] > 0}
} {1}

# Test 6: Error handling
test fold-6.1 {Error - missing arguments} -body {
    torch::fold
} -returnCodes error -match glob -result "*Usage*"

test fold-6.2 {Error - insufficient arguments} -body {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    torch::fold $input
} -returnCodes error -match glob -result "*Usage*"

test fold-6.3 {Error - invalid tensor name} -body {
    torch::fold "invalid_tensor" {2 2} {2 2}
} -returnCodes error -match glob -result "*Invalid input tensor*"

test fold-6.4 {Error - invalid output_size format} -body {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    torch::fold $input {3} {2 2}
} -returnCodes error -match glob -result "*Output size must be list of 2 ints*"

test fold-6.5 {Error - invalid kernel_size format} -body {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    torch::fold $input {2 2} {2}
} -returnCodes error -match glob -result "*Kernel size must be list of 2 ints*"

test fold-6.6 {Error - unknown named parameter} -body {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    torch::fold -input $input -output_size {2 2} -kernel_size {2 2} -unknown_param 1
} -returnCodes error -match glob -result "*Unknown parameter*"

test fold-6.7 {Error - missing value for parameter} -body {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    torch::fold -input $input -output_size {2 2} -kernel_size
} -returnCodes error -match glob -result "*Missing value for parameter*"

test fold-6.8 {Error - missing required parameters in named syntax} -body {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    torch::fold -input $input -output_size {2 2}
} -returnCodes error -match glob -result "*Required parameters missing*"

# Test 7: Mathematical correctness and tensor properties
test fold-7.1 {Verify result is a valid tensor} {
    # Create a tensor with the right shape for folding
    set data {}
    for {set i 1} {$i <= 12} {incr i} {
        lappend data $i.0
    }
    set input [torch::tensorCreate -data $data -shape {1 4 3} -dtype float32]
    set result [torch::fold $input {2 4} {2 2}]
    set shape [torch::tensor_size $result]
    llength $shape
} {4}

test fold-7.2 {Check output shape with basic folding} {
    # Create a tensor with the right shape for folding
    set data {}
    for {set i 1} {$i <= 12} {incr i} {
        lappend data $i.0
    }
    set input [torch::tensorCreate -data $data -shape {1 4 3} -dtype float32]
    set result [torch::fold $input {2 4} {2 2}]
    set shape [torch::tensor_size $result]
    list [lindex $shape 0] [lindex $shape 1]
} {1 1}

test fold-7.3 {Verify folding preserves data type} {
    # Create a tensor with the right shape for folding
    set data {}
    for {set i 1} {$i <= 12} {incr i} {
        lappend data $i.0
    }
    set input [torch::tensorCreate -data $data -shape {1 4 3} -dtype float32]
    set result [torch::fold $input {2 4} {2 2}]
    set dtype [torch::tensor_dtype $result]
    set input_dtype [torch::tensor_dtype $input]
    expr {$dtype eq $input_dtype}
} {1}

# Test 8: Syntax equivalence
test fold-8.1 {Positional and named syntax equivalence} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set result1 [torch::fold $input {2 2} {2 2}]
    set result2 [torch::fold -input $input -output_size {2 2} -kernel_size {2 2}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test fold-8.2 {Snake_case and camelCase parameter equivalence} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set result1 [torch::fold -input $input -output_size {2 2} -kernel_size {2 2}]
    set result2 [torch::fold -input $input -outputSize {2 2} -kernelSize {2 2}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test fold-8.3 {Command name aliases equivalence} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set result1 [torch::fold $input {2 2} {2 2}]
    set result2 [torch::Fold $input {2 2} {2 2}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Test 9: Integration
test fold-9.1 {Chain with other operations} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set folded [torch::fold $input {2 2} {2 2}]
    set summed [torch::tensor_sum $folded]
    expr {[string length $summed] > 0}
} {1}

test fold-9.2 {Multiple fold operations} {
    set input1 [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {1 4 1} -dtype float32]
    set input2 [torch::tensorCreate -data {2.0 4.0 6.0 8.0} -shape {1 4 1} -dtype float32]
    set result1 [torch::fold $input1 {2 2} {2 2}]
    set result2 [torch::fold $input2 {2 2} {2 2}]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

cleanupTests
