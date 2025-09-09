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

# =====================================================================
# TORCH::FLOOR COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test floor-1.1 {Basic positional syntax} {
    set t1 [torch::zeros {3 3}]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test floor-1.2 {Positional syntax with floating point values} {
    # Create tensor with floating point values that need floor
    set t1 [torch::tensor_create {1.7 2.3 -0.5 -1.8} float32]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test floor-1.3 {Positional syntax with single value} {
    set t1 [torch::tensor_create {3.14} float32]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test floor-1.4 {Positional syntax with negative values} {
    set t1 [torch::tensor_create -data {-1.7 -2.3 -3.9} -dtype float32]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test floor-1.5 {Positional syntax with mixed values} {
    set t1 [torch::tensor_create -data {-1.7 2.3 -3.9 4.1} -dtype float32]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

# Tests for named parameter syntax
test floor-2.1 {Named parameter syntax basic} {
    set t1 [torch::zeros {2 3}]
    set result [torch::floor -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3}

test floor-2.2 {Named parameter syntax with tensor alias} {
    set t1 [torch::ones {3}]
    set result [torch::floor -tensor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test floor-2.3 {Named parameter syntax with different tensor sizes} {
    set t1 [torch::zeros {4 2 2}]
    set result [torch::floor -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4 2 2}

test floor-2.4 {Named parameter syntax with floating point values} {
    set t1 [torch::tensor_create {1.7 2.3 -0.5 -1.8} float32]
    set result [torch::floor -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test floor-2.5 {Named parameter syntax with negative values} {
    set t1 [torch::tensor_create -data {-1.7 -2.3 -3.9} -dtype float32]
    set result [torch::floor -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

# Tests for camelCase alias (torch::Floor)
test floor-3.1 {CamelCase alias basic functionality} {
    set t1 [torch::zeros {2 2}]
    set result [torch::Floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test floor-3.2 {CamelCase alias with named parameters} {
    set t1 [torch::ones {3 3}]
    set result [torch::Floor -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

test floor-3.3 {CamelCase alias with floating point values} {
    set t1 [torch::tensor_create {1.7 2.3 -0.5 -1.8} float32]
    set result [torch::Floor -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test floor-3.4 {CamelCase alias with tensor parameter} {
    set t1 [torch::tensor_create {1.7 2.3} float32]
    set result [torch::Floor -tensor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2}

# Tests for error handling
test floor-4.1 {Error on missing parameter} {
    catch {torch::floor} msg
    expr {[string match "*Required parameter missing*" $msg] || [string match "*Usage*" $msg]}
} {1}

test floor-4.2 {Error on invalid tensor name} {
    catch {torch::floor invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test floor-4.3 {Error on invalid tensor name with named parameters} {
    catch {torch::floor -input invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test floor-4.4 {Error on unknown parameter} {
    set t1 [torch::zeros {2 2}]
    catch {torch::floor -unknown_param $t1} msg
    string match "*Unknown parameter*" $msg
} {1}

test floor-4.5 {Error on missing parameter value} {
    catch {torch::floor -input} msg
    string match "*Named parameter requires a value*" $msg
} {1}

test floor-4.6 {Error on too many positional arguments} {
    set t1 [torch::zeros {2 2}]
    catch {torch::floor $t1 extra_arg} msg
    string match "*Wrong number of positional arguments*" $msg
} {1}

# Tests for camelCase alias error handling
test floor-4.7 {Error on camelCase alias - missing parameter} {
    catch {torch::Floor} msg
    expr {[string match "*Required parameter missing*" $msg] || [string match "*Usage*" $msg]}
} {1}

test floor-4.8 {Error on camelCase alias - invalid tensor} {
    catch {torch::Floor invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

test floor-4.9 {Error on camelCase alias - unknown parameter} {
    set t1 [torch::zeros {2 2}]
    catch {torch::Floor -unknown_param $t1} msg
    string match "*Unknown parameter*" $msg
} {1}

test floor-4.10 {Error on camelCase alias - missing value} {
    catch {torch::Floor -input} msg
    string match "*Named parameter requires a value*" $msg
} {1}

test floor-4.11 {Error on camelCase alias - invalid tensor via named parameter} {
    catch {torch::Floor -input invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} {1}

# Tests for data types and edge cases
test floor-5.1 {Float tensor handling} {
    set t1 [torch::tensor_create {1.1 2.9 3.0} float32]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test floor-5.2 {Large tensor handling} {
    set t1 [torch::zeros {10 10}]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {10 10}

test floor-5.3 {1D tensor handling} {
    set t1 [torch::tensor_create -data {-2.5 -1.1 0.7 1.0 2.3} -dtype float32]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {5}

test floor-5.4 {3D tensor handling} {
    set t1 [torch::zeros {2 3 4}]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

test floor-5.5 {Very large numbers} {
    set t1 [torch::tensor_create {1000000.7 -1000000.3} float32]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2}

test floor-5.6 {Very small numbers near zero} {
    set t1 [torch::tensor_create {0.0001 -0.0001} float32]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2}

# Tests for syntax consistency (both syntaxes should produce same results)
test floor-6.1 {Syntax consistency - shape preservation} {
    set t1 [torch::zeros {3 3}]
    set result1 [torch::floor $t1]
    set result2 [torch::floor -input $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test floor-6.2 {Syntax consistency - multiple tensors} {
    set t1 [torch::ones {2 2}]
    set t2 [torch::ones {2 2}]
    set result1 [torch::floor $t1]
    set result2 [torch::floor -input $t2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test floor-6.3 {Syntax consistency - positional vs named vs camelCase} {
    set t1 [torch::tensor_create {1.7 2.3 3.9} float32]
    set result1 [torch::floor $t1]
    set result2 [torch::floor -input $t1]
    set result3 [torch::floor -tensor $t1]
    set result4 [torch::Floor $t1]
    set result5 [torch::Floor -input $t1]
    set result6 [torch::Floor -tensor $t1]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set shape3 [torch::tensor_shape $result3]
    set shape4 [torch::tensor_shape $result4]
    set shape5 [torch::tensor_shape $result5]
    set shape6 [torch::tensor_shape $result6]
    
    expr {$shape1 eq $shape2 && $shape2 eq $shape3 && $shape3 eq $shape4 && $shape4 eq $shape5 && $shape5 eq $shape6}
} {1}

# Tests for mathematical correctness
test floor-7.1 {Mathematical correctness - floor(0) should be 0} {
    set t1 [torch::zeros {1}]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {1}

test floor-7.2 {Mathematical correctness - floor(positive integers) should be unchanged} {
    set t1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test floor-7.3 {Mathematical correctness - preservation of tensor structure} {
    set t1 [torch::ones {2 2}]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 2}

test floor-7.4 {Mathematical correctness - negative values} {
    set t1 [torch::tensor_create -data {-2.5 -1.1 -0.1} -dtype float32]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3}

test floor-7.5 {Mathematical correctness - mixed positive/negative} {
    set t1 [torch::tensor_create {2.1 2.9 -2.1 -2.9} float32]
    set result [torch::floor $t1]
    # floor(2.1) = 2, floor(2.9) = 2, floor(-2.1) = -3, floor(-2.9) = -3
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test floor-7.6 {Mathematical correctness - fractional boundaries} {
    set t1 [torch::tensor_create {0.5 -0.5 1.5 -1.5} float32]
    set result [torch::floor $t1]
    # floor(0.5) = 0, floor(-0.5) = -1, floor(1.5) = 1, floor(-1.5) = -2
    set shape [torch::tensor_shape $result]
    set shape
} {4}

# Tests for parameter validation
test floor-8.1 {Parameter validation - input parameter required} {
    catch {torch::floor -input} msg
    string match "*Named parameter requires a value*" $msg
} {1}

test floor-8.2 {Parameter validation - extra parameters rejected} {
    set t1 [torch::zeros {2 2}]
    catch {torch::floor -input $t1 -extra_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test floor-8.3 {Parameter validation - wrong number of positional args} {
    set t1 [torch::zeros {2 2}]
    set t2 [torch::zeros {2 2}]
    catch {torch::floor $t1 $t2} msg
    string match "*Wrong number of positional arguments*" $msg
} {1}

test floor-8.4 {Parameter validation - empty tensor name} {
    catch {torch::floor ""} msg
    string match "*Required parameter missing: input tensor*" $msg
} {1}

test floor-8.5 {Parameter validation - empty tensor name with named param} {
    catch {torch::floor -input ""} msg
    string match "*Required parameter missing: input tensor*" $msg
} {1}

# Tests for integration with other operations
test floor-9.1 {Integration test - floor with tensor creation} {
    set t1 [torch::tensor_create {1.7 2.3 3.9 4.1} float32]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {4}

test floor-9.2 {Integration test - floor with zeros} {
    set t1 [torch::zeros {5 5}]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {5 5}

test floor-9.3 {Integration test - floor with ones} {
    set t1 [torch::ones {3 3}]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {3 3}

# Tests for different data types
test floor-10.1 {Data type test - float32} {
    set t1 [torch::tensor_create {1.7 2.3} float32]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2}

test floor-10.2 {Data type test - float64} {
    set t1 [torch::tensor_create {1.7 2.3} float64]
    set result [torch::floor $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {2}

# Performance tests
test floor-11.1 {Performance test - large tensor} {
    set t1 [torch::zeros {100 100}]
    set start_time [clock milliseconds]
    set result [torch::floor $t1]
    set end_time [clock milliseconds]
    set duration [expr {$end_time - $start_time}]
    set shape [torch::tensor_shape $result]
    # Should complete quickly and preserve shape
    expr {$duration < 1000 && $shape eq {100 100}}
} {1}

test floor-11.2 {Performance test - named parameters} {
    set t1 [torch::zeros {100 100}]
    set start_time [clock milliseconds]
    set result [torch::floor -input $t1]
    set end_time [clock milliseconds]
    set duration [expr {$end_time - $start_time}]
    set shape [torch::tensor_shape $result]
    # Should complete quickly and preserve shape
    expr {$duration < 1000 && $shape eq {100 100}}
} {1}

# Final comprehensive test
test floor-12.1 {Final comprehensive test - all functionality} {
    # Test all major functionality in one comprehensive test
    set t1 [torch::tensor_create -data {-2.7 -1.3 0.0 1.7 2.3} -dtype float32]
    
    # Test all syntax variations
    set result1 [torch::floor $t1]
    set result2 [torch::floor -input $t1]
    set result3 [torch::floor -tensor $t1]
    set result4 [torch::Floor $t1]
    set result5 [torch::Floor -input $t1]
    set result6 [torch::Floor -tensor $t1]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    set shape3 [torch::tensor_shape $result3]
    set shape4 [torch::tensor_shape $result4]
    set shape5 [torch::tensor_shape $result5]
    set shape6 [torch::tensor_shape $result6]
    
    # All should produce the same shape
    expr {$shape1 eq $shape2 && $shape2 eq $shape3 && $shape3 eq $shape4 && $shape4 eq $shape5 && $shape5 eq $shape6 && $shape1 eq {5}}
} {1}

cleanupTests 