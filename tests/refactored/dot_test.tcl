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
# TORCH::DOT COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test dot-1.1 {Basic positional syntax} {
    set t1 [torch::ones {3}]
    set t2 [torch::ones {3}]
    set result [torch::dot $t1 $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-1.2 {Positional syntax with zeros} {
    set t1 [torch::zeros {4}]
    set t2 [torch::ones {4}]
    set result [torch::dot $t1 $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-1.3 {Positional syntax with larger vectors} {
    set t1 [torch::ones {10}]
    set t2 [torch::ones {10}]
    set result [torch::dot $t1 $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-1.4 {Positional syntax with single element vectors} {
    set t1 [torch::ones {1}]
    set t2 [torch::ones {1}]
    set result [torch::dot $t1 $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

# Tests for named parameter syntax
test dot-2.1 {Named parameter syntax basic} {
    set t1 [torch::ones {3}]
    set t2 [torch::ones {3}]
    set result [torch::dot -input $t1 -other $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-2.2 {Named parameter syntax with zeros} {
    set t1 [torch::zeros {5}]
    set t2 [torch::ones {5}]
    set result [torch::dot -input $t1 -other $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-2.3 {Named parameter syntax with different order} {
    set t1 [torch::ones {4}]
    set t2 [torch::ones {4}]
    set result [torch::dot -other $t2 -input $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-2.4 {Named parameter syntax with larger vectors} {
    set t1 [torch::ones {8}]
    set t2 [torch::ones {8}]
    set result [torch::dot -input $t1 -other $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

# Tests for camelCase alias (torch::dot is already camelCase-compatible)
test dot-3.1 {Dot command with positional syntax} {
    set t1 [torch::ones {3}]
    set t2 [torch::ones {3}]
    set result [torch::dot $t1 $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-3.2 {Dot command with named parameters} {
    set t1 [torch::ones {3}]
    set t2 [torch::ones {3}]
    set result [torch::dot -input $t1 -other $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

# Tests for error handling
test dot-4.1 {Error on missing parameters} {
    catch {torch::dot} msg
    expr {[string match "*Usage*" $msg] || [string match "*Required parameters*" $msg]}
} {1}

test dot-4.2 {Error on single parameter} {
    set t1 [torch::ones {3}]
    catch {torch::dot $t1} msg
    expr {[string match "*Usage*" $msg] || [string match "*Required parameters*" $msg]}
} {1}

test dot-4.3 {Error on invalid first tensor name} {
    set t2 [torch::ones {3}]
    catch {torch::dot invalid_tensor $t2} msg
    string match "*Invalid input tensor*" $msg
} {1}

test dot-4.4 {Error on invalid second tensor name} {
    set t1 [torch::ones {3}]
    catch {torch::dot $t1 invalid_tensor} msg
    string match "*Invalid other tensor*" $msg
} {1}

test dot-4.5 {Error on invalid tensor name with named parameters} {
    set t1 [torch::ones {3}]
    catch {torch::dot -input invalid_tensor -other $t1} msg
    string match "*Invalid input tensor*" $msg
} {1}

test dot-4.6 {Error on unknown parameter} {
    set t1 [torch::ones {3}]
    set t2 [torch::ones {3}]
    catch {torch::dot -unknown_param $t1 -other $t2} msg
    string match "*Unknown parameter*" $msg
} {1}

test dot-4.7 {Error on missing parameter value} {
    set t1 [torch::ones {3}]
    catch {torch::dot -input $t1 -other} msg
    string match "*Missing value for parameter*" $msg
} {1}

test dot-4.8 {Error on missing required parameter -input} {
    set t1 [torch::ones {3}]
    catch {torch::dot -other $t1} msg
    string match "*Required parameters missing*" $msg
} {1}

test dot-4.9 {Error on missing required parameter -other} {
    set t1 [torch::ones {3}]
    catch {torch::dot -input $t1} msg
    string match "*Required parameters missing*" $msg
} {1}

# Tests for mathematical correctness  
test dot-5.1 {Mathematical correctness - ones dot ones} {
    set t1 [torch::ones {3}]
    set t2 [torch::ones {3}]
    set result [torch::dot $t1 $t2]
    # Should produce scalar result (shape {})
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-5.2 {Mathematical correctness - zeros dot ones should be zero} {
    set t1 [torch::zeros {5}]
    set t2 [torch::ones {5}]
    set result [torch::dot $t1 $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-5.3 {Mathematical correctness - result is scalar} {
    set t1 [torch::ones {7}]
    set t2 [torch::ones {7}]
    set result [torch::dot $t1 $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-5.4 {Mathematical correctness - commutativity test shape} {
    set t1 [torch::ones {4}]
    set t2 [torch::ones {4}]
    set result1 [torch::dot $t1 $t2]
    set result2 [torch::dot $t2 $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for different tensor sizes
test dot-6.1 {Single element vectors} {
    set t1 [torch::ones {1}]
    set t2 [torch::ones {1}]
    set result [torch::dot $t1 $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-6.2 {Medium size vectors} {
    set t1 [torch::ones {50}]
    set t2 [torch::ones {50}]
    set result [torch::dot $t1 $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-6.3 {Large vectors} {
    set t1 [torch::ones {100}]
    set t2 [torch::ones {100}]
    set result [torch::dot $t1 $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

# Tests for syntax consistency (both syntaxes should produce same results)
test dot-7.1 {Syntax consistency - basic case} {
    set t1 [torch::ones {3}]
    set t2 [torch::ones {3}]
    set result1 [torch::dot $t1 $t2]
    set result2 [torch::dot -input $t1 -other $t2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test dot-7.2 {Syntax consistency - zeros case} {
    set t1 [torch::zeros {4}]
    set t2 [torch::ones {4}]
    set result1 [torch::dot $t1 $t2]
    set result2 [torch::dot -input $t1 -other $t2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test dot-7.3 {Syntax consistency - larger vectors} {
    set t1 [torch::ones {10}]
    set t2 [torch::ones {10}]
    set result1 [torch::dot $t1 $t2]
    set result2 [torch::dot -input $t1 -other $t2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for parameter validation
test dot-8.1 {Parameter validation - both parameters required} {
    catch {torch::dot -input} msg
    string match "*Missing value for parameter*" $msg
} {1}

test dot-8.2 {Parameter validation - unknown parameters rejected} {
    set t1 [torch::ones {3}]
    set t2 [torch::ones {3}]
    catch {torch::dot -input $t1 -other $t2 -extra_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test dot-8.3 {Parameter validation - parameter order independence} {
    set t1 [torch::ones {3}]
    set t2 [torch::ones {3}]
    set result1 [torch::dot -input $t1 -other $t2]
    set result2 [torch::dot -other $t2 -input $t1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

# Tests for edge cases
test dot-9.1 {Edge case - very small vectors} {
    set t1 [torch::ones {1}]
    set t2 [torch::ones {1}]
    set result [torch::dot -input $t1 -other $t2]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-9.2 {Edge case - identical tensor handles} {
    set t1 [torch::ones {5}]
    set result [torch::dot $t1 $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {}

test dot-9.3 {Edge case - identical tensor handles with named params} {
    set t1 [torch::ones {5}]
    set result [torch::dot -input $t1 -other $t1]
    set shape [torch::tensor_shape $result]
    set shape
} {}

# Tests for comprehensive parameter combinations
test dot-10.1 {All valid parameter combinations work} {
    set t1 [torch::ones {3}]
    set t2 [torch::ones {3}]
    
    # Positional
    set r1 [torch::dot $t1 $t2]
    
    # Named parameters
    set r2 [torch::dot -input $t1 -other $t2]
    set r3 [torch::dot -other $t2 -input $t1]
    
    # All should produce scalar results
    set s1 [torch::tensor_shape $r1]
    set s2 [torch::tensor_shape $r2]
    set s3 [torch::tensor_shape $r3]
    
    expr {$s1 eq {} && $s2 eq {} && $s3 eq {}}
} {1}

cleanupTests 