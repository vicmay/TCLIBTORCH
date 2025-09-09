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
# TORCH::DISTRIBUTED_GATHER Tests - Dual Syntax Support
# ============================================================================

# Test 1: Basic positional syntax (backward compatibility)
test distributed_gather-1.1 {Basic positional syntax} {
    set tensor [torch::zeros {2 3} float32]
    set result [torch::distributed_gather $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 3}

test distributed_gather-1.2 {Positional syntax with dst parameter} {
    set tensor [torch::ones {2 2} float32]
    set result [torch::distributed_gather $tensor 0]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 2}

test distributed_gather-1.3 {Positional syntax with dst and group parameters} {
    set tensor [torch::ones {3 3} float32]
    set result [torch::distributed_gather $tensor 0 "default"]
    set shape [torch::tensor_shape $result]
    set shape
} {1 3 3}

# Test 2: Named parameter syntax
test distributed_gather-2.1 {Named parameter syntax - basic} {
    set tensor [torch::zeros {2 3} float32]
    set result [torch::distributed_gather -tensor $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 3}

test distributed_gather-2.2 {Named parameter syntax with dst} {
    set tensor [torch::ones {2 2} float32]
    set result [torch::distributed_gather -tensor $tensor -dst 1]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 2}

test distributed_gather-2.3 {Named parameter syntax with all parameters} {
    set tensor [torch::ones {3 3} float32]
    set result [torch::distributed_gather -tensor $tensor -dst 0 -group "default"]
    set shape [torch::tensor_shape $result]
    set shape
} {1 3 3}

test distributed_gather-2.4 {Named parameter syntax - different parameter order} {
    set tensor [torch::ones {2 2} float32]
    set result [torch::distributed_gather -dst 1 -tensor $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 2}

test distributed_gather-2.5 {Named parameter syntax with group only} {
    set tensor [torch::zeros {2 3} float32]
    set result [torch::distributed_gather -tensor $tensor -group "test_group"]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 3}

# Test 3: camelCase alias tests
test distributed_gather-3.1 {camelCase alias - basic} {
    set tensor [torch::zeros {2 3} float32]
    set result [torch::distributedGather -tensor $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 3}

test distributed_gather-3.2 {camelCase alias with all parameters} {
    set tensor [torch::ones {3 3} float32]
    set result [torch::distributedGather -tensor $tensor -dst 1 -group "test"]
    set shape [torch::tensor_shape $result]
    set shape
} {1 3 3}

test distributed_gather-3.3 {camelCase alias positional syntax} {
    set tensor [torch::ones {2 2} float32]
    set result [torch::distributedGather $tensor 0]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 2}

# Test 4: Data type variations
test distributed_gather-4.1 {Different data types - float64} {
    set tensor [torch::zeros {2 2} float64]
    set result [torch::distributed_gather -tensor $tensor]
    torch::tensor_dtype $result
} {Float64}

test distributed_gather-4.2 {Different data types - int32} {
    set tensor [torch::ones {2 2} int32]
    set result [torch::distributed_gather -tensor $tensor]
    torch::tensor_dtype $result
} {Int32}

test distributed_gather-4.3 {Different data types - int64} {
    set tensor [torch::zeros {3 3} int64]
    set result [torch::distributed_gather -tensor $tensor]
    torch::tensor_dtype $result
} {Int64}

# Test 5: Shape variations
test distributed_gather-5.1 {1D tensor} {
    set tensor [torch::ones {5} float32]
    set result [torch::distributed_gather -tensor $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {1 5}

test distributed_gather-5.2 {3D tensor} {
    set tensor [torch::zeros {2 3 4} float32]
    set result [torch::distributed_gather -tensor $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 3 4}

test distributed_gather-5.3 {4D tensor} {
    set tensor [torch::ones {2 2 2 2} float32]
    set result [torch::distributed_gather -tensor $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 2 2 2}

# Test 6: Edge cases
test distributed_gather-6.1 {Single element tensor} {
    set tensor [torch::tensor_create {5.0} float32]
    set result [torch::distributed_gather -tensor $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1}

test distributed_gather-6.2 {Large tensor} {
    set tensor [torch::zeros {10 10} float32]
    set result [torch::distributed_gather -tensor $tensor]
    set shape [torch::tensor_shape $result]
    set shape
} {1 10 10}

test distributed_gather-6.3 {Different dst values} {
    set tensor [torch::ones {2 2} float32]
    set result1 [torch::distributed_gather -tensor $tensor -dst 0]
    set result2 [torch::distributed_gather -tensor $tensor -dst 1]
    set result3 [torch::distributed_gather -tensor $tensor -dst -1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]  
    set shape3 [torch::tensor_shape $result3]
    expr {$shape1 eq "1 2 2" && $shape2 eq "1 2 2" && $shape3 eq "1 2 2"}
} {1}

# Test 7: Error handling tests
test distributed_gather-7.1 {Missing tensor parameter in named syntax} {
    catch {torch::distributed_gather -dst 0} result
    expr {[string match "*Required parameter missing: -tensor*" $result]}
} {1}

test distributed_gather-7.2 {Invalid tensor name} {
    catch {torch::distributed_gather -tensor "invalid_tensor"} result
    expr {[string match "*Invalid tensor name*" $result]}
} {1}

test distributed_gather-7.3 {Invalid dst parameter type} {
    set tensor [torch::ones -shape {2 2} -dtype float32]
    catch {torch::distributed_gather -tensor $tensor -dst "not_a_number"} result
    expr {[string match "*Invalid -dst parameter*" $result]}
} {1}

test distributed_gather-7.4 {Unknown parameter} {
    set tensor [torch::ones -shape {2 2} -dtype float32]
    catch {torch::distributed_gather -tensor $tensor -unknown_param value} result
    expr {[string match "*Unknown parameter: -unknown_param*" $result]}
} {1}

test distributed_gather-7.5 {Missing value for parameter} {
    set tensor [torch::ones -shape {2 2} -dtype float32]
    catch {torch::distributed_gather -tensor $tensor -dst} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

test distributed_gather-7.6 {Wrong number of positional arguments - too few} {
    catch {torch::distributed_gather} result
    expr {[string match "*Required parameter missing*" $result] || [string match "*Wrong number of arguments*" $result]}
} {1}

test distributed_gather-7.7 {Wrong number of positional arguments - too many} {
    set tensor [torch::ones -shape {2 2} -dtype float32]
    catch {torch::distributed_gather $tensor 0 "group" extra_arg} result
    expr {[string match "*Wrong number of arguments*" $result]}
} {1}

# Test 8: Consistency between syntaxes
test distributed_gather-8.1 {Positional vs named consistency - basic} {
    set tensor [torch::ones {2 3} float32]
    set result_pos [torch::distributed_gather $tensor]
    set result_named [torch::distributed_gather -tensor $tensor]
    set shape_pos [torch::tensor_shape $result_pos]
    set shape_named [torch::tensor_shape $result_named]
    expr {$shape_pos eq $shape_named}
} {1}

test distributed_gather-8.2 {Positional vs named consistency - with dst} {
    set tensor [torch::ones {2 3} float32]
    set result_pos [torch::distributed_gather $tensor 1]
    set result_named [torch::distributed_gather -tensor $tensor -dst 1]
    set shape_pos [torch::tensor_shape $result_pos]
    set shape_named [torch::tensor_shape $result_named]
    expr {$shape_pos eq $shape_named}
} {1}

test distributed_gather-8.3 {Positional vs named consistency - full parameters} {
    set tensor [torch::ones {2 3} float32]
    set result_pos [torch::distributed_gather $tensor 0 "test_group"]
    set result_named [torch::distributed_gather -tensor $tensor -dst 0 -group "test_group"]
    set shape_pos [torch::tensor_shape $result_pos]
    set shape_named [torch::tensor_shape $result_named]
    expr {$shape_pos eq $shape_named}
} {1}

test distributed_gather-8.4 {snake_case vs camelCase consistency} {
    set tensor [torch::ones {2 3} float32]
    set result_snake [torch::distributed_gather -tensor $tensor -dst 1]
    set result_camel [torch::distributedGather -tensor $tensor -dst 1]
    set shape_snake [torch::tensor_shape $result_snake]
    set shape_camel [torch::tensor_shape $result_camel]
    expr {$shape_snake eq $shape_camel}
} {1}

# Test 9: Parameter validation
test distributed_gather-9.1 {Valid dst parameter - zero} {
    set tensor [torch::ones {2 2} float32]
    set result [torch::distributed_gather -tensor $tensor -dst 0]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 2}

test distributed_gather-9.2 {Valid dst parameter - positive} {
    set tensor [torch::ones {2 2} float32]
    set result [torch::distributed_gather -tensor $tensor -dst 5]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 2}

test distributed_gather-9.3 {Valid dst parameter - negative} {
    set tensor [torch::ones {2 2} float32]
    set result [torch::distributed_gather -tensor $tensor -dst -1]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 2}

test distributed_gather-9.4 {Empty group parameter} {
    set tensor [torch::ones {2 2} float32]
    set result [torch::distributed_gather -tensor $tensor -group ""]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 2}

test distributed_gather-9.5 {Group parameter with spaces} {
    set tensor [torch::ones {2 2} float32]
    set result [torch::distributed_gather -tensor $tensor -group "test group"]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 2}

# Test 10: Complex scenarios
test distributed_gather-10.1 {Multiple gather operations} {
    set tensor1 [torch::ones {2 2} float32]
    set tensor2 [torch::zeros {3 3} float32]
    set result1 [torch::distributed_gather -tensor $tensor1]
    set result2 [torch::distributed_gather -tensor $tensor2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq "1 2 2" && $shape2 eq "1 3 3"}
} {1}

test distributed_gather-10.2 {Mixed syntax usage} {
    set tensor [torch::ones {2 3} float32]
    set result1 [torch::distributed_gather $tensor 0]
    set result2 [torch::distributedGather -tensor $tensor -dst 0]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test distributed_gather-10.3 {Gather with computation chain} {
    set base [torch::zeros {2 2} float32]
    set ones [torch::ones {2 2} float32]
    set processed [torch::tensor_add $base $ones]
    set result [torch::distributed_gather -tensor $processed]
    set shape [torch::tensor_shape $result]
    set shape
} {1 2 2}

cleanupTests 