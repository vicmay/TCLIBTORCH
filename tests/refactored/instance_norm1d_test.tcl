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

# ============================================================================
# Test torch::instance_norm1d - 1D Instance Normalization
# ============================================================================

# Test 1: Basic functionality with positional syntax
test instance_norm1d-1.1 {Basic positional syntax - default parameters} {
    set tensor [torch::ones -shape {2 3 4}]
    set result [torch::instance_norm1d $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

test instance_norm1d-1.2 {Positional syntax with eps parameter} {
    set tensor [torch::ones -shape {1 5 8}]
    set result [torch::instance_norm1d $tensor 1e-4]
    set shape [torch::tensor_shape $result]
    set shape
} {1 5 8}

test instance_norm1d-1.3 {Positional syntax with eps and momentum} {
    set tensor [torch::ones -shape {2 4 6}]
    set result [torch::instance_norm1d $tensor 1e-5 0.2]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4 6}

# Test 2: Named parameter syntax
test instance_norm1d-2.1 {Named parameter syntax - basic} {
    set tensor [torch::ones -shape {2 3 4}]
    set result [torch::instance_norm1d -input $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

test instance_norm1d-2.2 {Named parameter syntax - alternative input parameter} {
    set tensor [torch::ones -shape {2 3 4}]
    set result [torch::instance_norm1d -tensor $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

test instance_norm1d-2.3 {Named parameter syntax with eps} {
    set tensor [torch::ones -shape {1 5 8}]
    set result [torch::instance_norm1d -input $tensor -eps 1e-4]
    set shape [torch::tensor_shape $result]
    set shape
} {1 5 8}

test instance_norm1d-2.4 {Named parameter syntax with epsilon alias} {
    set tensor [torch::ones -shape {1 5 8}]
    set result [torch::instance_norm1d -input $tensor -epsilon 1e-4]
    set shape [torch::tensor_shape $result]
    set shape
} {1 5 8}

test instance_norm1d-2.5 {Named parameter syntax with momentum} {
    set tensor [torch::ones -shape {2 4 6}]
    set result [torch::instance_norm1d -input $tensor -momentum 0.2]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4 6}

test instance_norm1d-2.6 {Named parameter syntax with affine} {
    set tensor [torch::ones -shape {2 3 4}]
    set result [torch::instance_norm1d -input $tensor -affine 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

test instance_norm1d-2.7 {Named parameter syntax with track_running_stats} {
    set tensor [torch::ones -shape {2 3 4}]
    set result [torch::instance_norm1d -input $tensor -track_running_stats 0]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

test instance_norm1d-2.8 {Named parameter syntax with trackRunningStats alias} {
    set tensor [torch::ones -shape {2 3 4}]
    set result [torch::instance_norm1d -input $tensor -trackRunningStats 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

test instance_norm1d-2.9 {Named parameter syntax with all parameters} {
    set tensor [torch::ones -shape {2 4 6}]
    set result [torch::instance_norm1d -input $tensor -eps 1e-4 -momentum 0.15 -affine 1 -track_running_stats 1]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4 6}

# Test 3: CamelCase alias
test instance_norm1d-3.1 {CamelCase alias - basic functionality} {
    set tensor [torch::ones -shape {2 3 4}]
    set result [torch::instanceNorm1d $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

test instance_norm1d-3.2 {CamelCase alias with named parameters} {
    set tensor [torch::ones -shape {2 4 6}]
    set result [torch::instanceNorm1d -input $tensor -eps 1e-4]
    set shape [torch::tensor_shape $result]
    set shape
} {2 4 6}

# Test 4: Error handling
test instance_norm1d-4.1 {Error handling - missing input} {
    catch {torch::instance_norm1d} msg
    string match "*Required parameters missing*" $msg
} 1

test instance_norm1d-4.2 {Error handling - invalid eps} {
    set tensor [torch::ones -shape {2 3 4}]
    catch {torch::instance_norm1d $tensor invalid_eps} msg
    string match "*Invalid eps*" $msg
} 1

test instance_norm1d-4.3 {Error handling - invalid momentum} {
    set tensor [torch::ones -shape {2 3 4}]
    catch {torch::instance_norm1d $tensor 1e-5 invalid_momentum} msg
    string match "*Invalid momentum*" $msg
} 1

test instance_norm1d-4.4 {Error handling - unknown parameter} {
    set tensor [torch::ones -shape {2 3 4}]
    catch {torch::instance_norm1d -input $tensor -unknown_param value} msg
    string match "*Unknown parameter*" $msg
} 1

test instance_norm1d-4.5 {Error handling - missing parameter value} {
    set tensor [torch::ones -shape {2 3 4}]
    catch {torch::instance_norm1d -input $tensor -eps} msg
    string match "*Named parameters must come in pairs*" $msg
} 1

# Test 5: Consistency between syntaxes
test instance_norm1d-5.1 {Consistency - positional vs named basic} {
    set tensor [torch::ones -shape {2 3 4}]
    set result1 [torch::instance_norm1d $tensor]
    set result2 [torch::instance_norm1d -input $tensor]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test instance_norm1d-5.2 {Consistency - positional vs named with eps} {
    set tensor [torch::ones -shape {1 5 8}]
    set result1 [torch::instance_norm1d $tensor 1e-4]
    set result2 [torch::instance_norm1d -input $tensor -eps 1e-4]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test instance_norm1d-5.3 {Consistency - snake_case vs camelCase} {
    set tensor [torch::ones -shape {2 3 4}]
    set result1 [torch::instance_norm1d $tensor]
    set result2 [torch::instanceNorm1d $tensor]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

# Test 6: Different tensor types and shapes
test instance_norm1d-6.1 {Different tensor types - float32} {
    set tensor [torch::ones -shape {2 3 4} -dtype float32]
    set result [torch::instance_norm1d $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 4}

test instance_norm1d-6.2 {Different tensor types - int64 (expected error)} {
    set tensor [torch::ones -shape {2 3 4} -dtype int64]
    catch {torch::instance_norm1d $tensor} msg
    string match "*not implemented for*" $msg
} 1

test instance_norm1d-6.3 {Different shapes - larger tensor} {
    set tensor [torch::ones -shape {4 8 16}]
    set result [torch::instance_norm1d $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {4 8 16}

test instance_norm1d-6.4 {Different shapes - single batch} {
    set tensor [torch::ones -shape {1 10 20}]
    set result [torch::instance_norm1d $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {1 10 20}

# Test 7: Parameter validation
test instance_norm1d-7.1 {Parameter validation - eps must be positive} {
    set tensor [torch::ones -shape {2 3 4}]
    catch {torch::instance_norm1d -input $tensor -eps -1.0} msg
    string match "*eps*" $msg
} 1

test instance_norm1d-7.2 {Parameter validation - momentum must be non-negative} {
    set tensor [torch::ones -shape {2 3 4}]
    catch {torch::instance_norm1d -input $tensor -momentum -0.1} msg
    string match "*momentum*" $msg
} 1

test instance_norm1d-7.3 {Parameter validation - affine boolean values} {
    set tensor [torch::ones -shape {2 3 4}]
    set result1 [torch::instance_norm1d -input $tensor -affine 0]
    set result2 [torch::instance_norm1d -input $tensor -affine 1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test instance_norm1d-7.4 {Parameter validation - track_running_stats boolean values} {
    set tensor [torch::ones -shape {2 3 4}]
    set result1 [torch::instance_norm1d -input $tensor -track_running_stats 0]
    set result2 [torch::instance_norm1d -input $tensor -track_running_stats 1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

cleanupTests 