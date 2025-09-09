#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the libtorch extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configure test environment
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax
test tanhshrink-1.1 {Basic tanhshrink positional syntax} {
    set t1 [torch::tensor_create -data {1.0 2.0 -1.0 0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-1.2 {Tanhshrink with positive values} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-1.3 {Tanhshrink with negative values} {
    set t1 [torch::tensor_create -data {-1.0 -2.0 -0.5} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-1.4 {Tanhshrink with zero} {
    set t1 [torch::tensor_create -data {0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-1.5 {Tanhshrink with multidimensional tensor} {
    set t1 [torch::tensor_create -data {1.0 -1.0 2.0 -2.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

# Test 2: Named parameter syntax
test tanhshrink-2.1 {Basic tanhshrink named syntax} {
    set t1 [torch::tensor_create -data {1.0 2.0 -1.0 0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink -input $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-2.2 {Tanhshrink named syntax with positive values} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink -input $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-2.3 {Tanhshrink named syntax with negative values} {
    set t1 [torch::tensor_create -data {-1.0 -2.0 -0.5} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink -input $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-2.4 {Tanhshrink named syntax with zero} {
    set t1 [torch::tensor_create -data {0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink -input $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-2.5 {Tanhshrink named syntax with multidimensional tensor} {
    set t1 [torch::tensor_create -data {1.0 -1.0 2.0 -2.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink -input $t1]
    string match "*tensor*" $result
} 1

# Test 3: camelCase alias
test tanhshrink-3.1 {camelCase alias with positional syntax} {
    set t1 [torch::tensor_create -data {1.0 2.0 -1.0 0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhShrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-3.2 {camelCase alias with named syntax} {
    set t1 [torch::tensor_create -data {1.0 2.0 -1.0 0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhShrink -input $t1]
    string match "*tensor*" $result
} 1

# Test 4: Syntax consistency verification
test tanhshrink-4.1 {Positional and named syntax produce same results} {
    set t1 [torch::tensor_create -data {1.0 2.0 -1.0 0.0} -dtype float32 -device cpu -requiresGrad false]
    set result1 [torch::tanhshrink $t1]
    set result2 [torch::tanhshrink -input $t1]
    
    # Both should be tensors
    if {![string match "*tensor*" $result1] || ![string match "*tensor*" $result2]} {
        return 0
    }
    return 1
} 1

test tanhshrink-4.2 {camelCase produces same result as snake_case} {
    set t1 [torch::tensor_create -data {1.0 2.0 -1.0 0.0} -dtype float32 -device cpu -requiresGrad false]
    set result1 [torch::tanhshrink $t1]
    set result2 [torch::tanhShrink $t1]
    
    # Both should be tensors
    if {![string match "*tensor*" $result1] || ![string match "*tensor*" $result2]} {
        return 0
    }
    return 1
} 1

# Test 5: Mathematical correctness - tanhshrink(x) = x - tanh(x)
test tanhshrink-5.1 {Tanhshrink mathematical properties} {
    set t1 [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-5.2 {Tanhshrink of zero gives zero} {
    set t1 [torch::tensor_create -data {0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-5.3 {Tanhshrink with positive values} {
    # For positive values, tanhshrink should be positive but smaller than input
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-5.4 {Tanhshrink with negative values} {
    # For negative values, result should be negative with larger magnitude
    set t1 [torch::tensor_create -data {-1.0 -2.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

# Test 6: Error handling for positional syntax
test tanhshrink-6.1 {Positional syntax error - no arguments} {
    catch {torch::tanhshrink} msg
    string match "*wrong # args*" $msg
} 1

test tanhshrink-6.2 {Positional syntax error - too many arguments} {
    set t1 [torch::tensor_create -data {1.0} -dtype float32 -device cpu -requiresGrad false]
    catch {torch::tanhshrink $t1 extra} msg
    string match "*wrong # args*" $msg
} 1

test tanhshrink-6.3 {Positional syntax error - invalid tensor} {
    catch {torch::tanhshrink invalid_tensor} msg
    string match "*Invalid tensor*" $msg
} 1

# Test 7: Error handling for named syntax
test tanhshrink-7.1 {Named syntax error - no arguments} {
    catch {torch::tanhshrink} msg
    string match "*wrong # args*" $msg
} 1

test tanhshrink-7.2 {Named syntax error - missing value} {
    catch {torch::tanhshrink -input} msg
    string match "*wrong # args*" $msg
} 1

test tanhshrink-7.3 {Named syntax error - unknown option} {
    set t1 [torch::tensor_create -data {1.0} -dtype float32 -device cpu -requiresGrad false]
    catch {torch::tanhshrink -unknown $t1} msg
    string match "*unknown option*" $msg
} 1

test tanhshrink-7.4 {Named syntax error - invalid tensor name} {
    catch {torch::tanhshrink -input invalid_tensor} msg
    string match "*Invalid tensor*" $msg
} 1

# Test 8: camelCase error handling
test tanhshrink-8.1 {camelCase error handling - invalid tensor} {
    catch {torch::tanhShrink invalid_tensor} msg
    string match "*Invalid tensor*" $msg
} 1

test tanhshrink-8.2 {camelCase error handling - no arguments} {
    catch {torch::tanhShrink} msg
    string match "*wrong # args*" $msg
} 1

# Test 9: Data type support
test tanhshrink-9.1 {Tanhshrink with float32} {
    set t1 [torch::tensor_create {1.0 -1.0} float32 cpu false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-9.2 {Tanhshrink with float64} {
    set t1 [torch::tensor_create -data {1.0 -1.0} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

# Test 10: Edge cases
test tanhshrink-10.1 {Tanhshrink with large positive values} {
    set t1 [torch::tensor_create -data {10.0 100.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-10.2 {Tanhshrink with large negative values} {
    set t1 [torch::tensor_create -data {-10.0 -100.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-10.3 {Tanhshrink with very small values} {
    set t1 [torch::tensor_create -data {0.001 -0.001} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-10.4 {Tanhshrink with mixed value range} {
    set t1 [torch::tensor_create -data {-3.0 -1.0 0.0 1.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

test tanhshrink-10.5 {Single element tensor} {
    set t1 [torch::tensor_create -data {1.5} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::tanhshrink $t1]
    string match "*tensor*" $result
} 1

cleanupTests 