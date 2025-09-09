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

;# Create test tensors
set input_tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set input_tensor2 [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu true]
set single_tensor [torch::tensor_create {1.0} float32 cpu true]
set zero_tensor [torch::tensor_create {0.0 0.0 0.0 0.0} float32 cpu true]

;# Test cases for positional syntax
test tensor-irfft-1.1 {Basic positional syntax - tensor only} {
    set result [torch::tensor_irfft $input_tensor]
    list [torch::tensor_shape $result]
} {6}

test tensor-irfft-1.2 {Positional syntax with n parameter} {
    set result [torch::tensor_irfft $input_tensor 6]
    list [torch::tensor_shape $result]
} {6}

test tensor-irfft-1.3 {Positional syntax with n and dim parameters} {
    set result [torch::tensor_irfft $input_tensor 4 0]
    list [torch::tensor_shape $result]
} {4}

;# Test cases for named syntax
test tensor-irfft-2.1 {Named parameter syntax - tensor only} {
    set result [torch::tensor_irfft -tensor $input_tensor]
    list [torch::tensor_shape $result]
} {6}

test tensor-irfft-2.2 {Named syntax with -input alias} {
    set result [torch::tensor_irfft -input $input_tensor]
    list [torch::tensor_shape $result]
} {6}

test tensor-irfft-2.3 {Named syntax with n parameter} {
    set result [torch::tensor_irfft -tensor $input_tensor -n 6]
    list [torch::tensor_shape $result]
} {6}

test tensor-irfft-2.4 {Named syntax with all parameters} {
    set result [torch::tensor_irfft -tensor $input_tensor -n 4 -dim 0]
    list [torch::tensor_shape $result]
} {4}

;# Test cases for camelCase alias
test tensor-irfft-3.1 {CamelCase alias - basic usage} {
    set result [torch::tensorIrfft $input_tensor]
    list [torch::tensor_shape $result]
} {6}

test tensor-irfft-3.2 {CamelCase alias with named parameters} {
    set result [torch::tensorIrfft -tensor $input_tensor -n 6]
    list [torch::tensor_shape $result]
} {6}

;# Test cases for error handling
test tensor-irfft-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_irfft nonexistent} result
    string match "*Invalid tensor name*" $result
} {1}

test tensor-irfft-4.2 {Error handling - missing tensor parameter} {
    catch {torch::tensor_irfft} result
    string match "*Required tensor parameter missing*" $result
} {1}

test tensor-irfft-4.3 {Error handling - invalid n parameter} {
    catch {torch::tensor_irfft $input_tensor invalid} result
    string match "*Invalid n parameter*" $result
} {1}

test tensor-irfft-4.4 {Error handling - invalid dim parameter} {
    catch {torch::tensor_irfft $input_tensor 4 invalid} result
    string match "*Invalid dim parameter*" $result
} {1}

test tensor-irfft-4.5 {Error handling - unknown named parameter} {
    catch {torch::tensor_irfft -tensor $input_tensor -unknown value} result
    string match "*Unknown parameter*" $result
} {1}

test tensor-irfft-4.6 {Error handling - missing value for parameter} {
    catch {torch::tensor_irfft -tensor $input_tensor -n} result
    string match "*Missing value for parameter*" $result
} {1}

;# Test cases for edge cases
test tensor-irfft-5.1 {Edge case - single element tensor} {
    catch {torch::tensor_irfft $single_tensor} result
    string match "*Invalid number of data points*" $result
} {1}

test tensor-irfft-5.2 {Edge case - zero tensor} {
    set result [torch::tensor_irfft $zero_tensor]
    list [torch::tensor_shape $result]
} {6}

test tensor-irfft-5.3 {Edge case - large n value} {
    set result [torch::tensor_irfft $input_tensor 100]
    list [torch::tensor_shape $result]
} {100}

test tensor-irfft-5.4 {Edge case - negative dim} {
    set result [torch::tensor_irfft $input_tensor 4 -1]
    list [torch::tensor_shape $result]
} {4}

;# Test cases for syntax consistency
test tensor-irfft-6.1 {Syntax consistency - positional vs named} {
    set result1 [torch::tensor_irfft $input_tensor 6 0]
    set result2 [torch::tensor_irfft -tensor $input_tensor -n 6 -dim 0]
    list [torch::tensor_shape $result1] [torch::tensor_shape $result2]
} {6 6}

test tensor-irfft-6.2 {Syntax consistency - snake_case vs camelCase} {
    set result1 [torch::tensor_irfft $input_tensor 4]
    set result2 [torch::tensorIrfft $input_tensor 4]
    list [torch::tensor_shape $result1] [torch::tensor_shape $result2]
} {4 4}

;# Test cases for complex input
test tensor-irfft-7.1 {Complex input tensor} {
    set result [torch::tensor_irfft $input_tensor2 10]
    list [torch::tensor_shape $result]
} {10}

test tensor-irfft-7.2 {Complex input with different n values} {
    set result1 [torch::tensor_irfft $input_tensor2 4]
    set result2 [torch::tensor_irfft $input_tensor2 8]
    set result3 [torch::tensor_irfft $input_tensor2 16]
    list [torch::tensor_shape $result1] [torch::tensor_shape $result2] [torch::tensor_shape $result3]
} {4 8 16}

;# Cleanup after all tests
cleanupTests 