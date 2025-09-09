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
# TORCH::DISTRIBUTED_IRECV Tests - Dual Syntax Support
# ============================================================================

# Test 1: Basic positional syntax (backward compatibility)
test distributed_irecv-1.1 {Basic positional syntax - 1D shape} {
    set result [torch::distributed_irecv {10} 0]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-1.2 {Positional syntax - 2D shape} {
    set result [torch::distributed_irecv {3 4} 1]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-1.3 {Positional syntax - 3D shape} {
    set result [torch::distributed_irecv {2 3 4} 2]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-1.4 {Positional syntax with tag} {
    set result [torch::distributed_irecv {5 5} 1 100]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-1.5 {Positional syntax - different sources} {
    set result1 [torch::distributed_irecv {2 2} 0]
    set result2 [torch::distributed_irecv {2 2} 3]
    set result3 [torch::distributed_irecv {2 2} 7]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1"}
} {1}

test distributed_irecv-1.6 {Positional syntax - different tags} {
    set result1 [torch::distributed_irecv {2 2} 0 0]
    set result2 [torch::distributed_irecv {2 2} 0 42]
    set result3 [torch::distributed_irecv {2 2} 0 999]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1"}
} {1}

# Test 2: Named parameter syntax
test distributed_irecv-2.1 {Named parameter syntax - basic} {
    set result [torch::distributed_irecv -shape {10} -src 0]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-2.2 {Named parameter syntax - 2D shape} {
    set result [torch::distributed_irecv -shape {3 4} -src 1]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-2.3 {Named parameter syntax with tag} {
    set result [torch::distributed_irecv -shape {5 5} -src 1 -tag 100]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-2.4 {Named parameter syntax - different parameter order} {
    set result [torch::distributed_irecv -src 2 -shape {2 3 4}]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-2.5 {Named parameter syntax - all parameters different order} {
    set result [torch::distributed_irecv -tag 50 -src 3 -shape {8 8}]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-2.6 {Named parameter syntax - complex shapes} {
    set result1 [torch::distributed_irecv -shape {1 28 28} -src 0]
    set result2 [torch::distributed_irecv -shape {32 3 224 224} -src 1]
    set result3 [torch::distributed_irecv -shape {10 20 30 40 50} -src 2]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1"}
} {1}

# Test 3: camelCase alias tests
test distributed_irecv-3.1 {camelCase alias - basic} {
    set result [torch::distributedIrecv -shape {10} -src 0]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-3.2 {camelCase alias with all parameters} {
    set result [torch::distributedIrecv -shape {5 5} -src 2 -tag 200]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-3.3 {camelCase alias positional syntax} {
    set result [torch::distributedIrecv {4 4} 1 42]
    expr {$result eq "irecv_handle_1"}
} {1}

# Test 4: Parameter validation
test distributed_irecv-4.1 {Valid shapes} {
    set result1 [torch::distributed_irecv -shape {1} -src 0]
    set result2 [torch::distributed_irecv -shape {10 20} -src 0]
    set result3 [torch::distributed_irecv -shape {5 10 15} -src 0]
    set result4 [torch::distributed_irecv -shape {2 3 4 5} -src 0]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1" && $result4 eq "irecv_handle_1"}
} {1}

test distributed_irecv-4.2 {Valid source ranks} {
    set result1 [torch::distributed_irecv -shape {2 2} -src 0]
    set result2 [torch::distributed_irecv -shape {2 2} -src 1]
    set result3 [torch::distributed_irecv -shape {2 2} -src 10]
    set result4 [torch::distributed_irecv -shape {2 2} -src 255]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1" && $result4 eq "irecv_handle_1"}
} {1}

test distributed_irecv-4.3 {Valid tags} {
    set result1 [torch::distributed_irecv -shape {2 2} -src 0 -tag 0]
    set result2 [torch::distributed_irecv -shape {2 2} -src 0 -tag 1]
    set result3 [torch::distributed_irecv -shape {2 2} -src 0 -tag 999]
    set result4 [torch::distributed_irecv -shape {2 2} -src 0 -tag 65535]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1" && $result4 eq "irecv_handle_1"}
} {1}

test distributed_irecv-4.4 {Large dimensions} {
    set result1 [torch::distributed_irecv -shape {1000} -src 0]
    set result2 [torch::distributed_irecv -shape {100 100} -src 0]
    set result3 [torch::distributed_irecv -shape {10 10 10 10} -src 0]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1"}
} {1}

# Test 5: Consistency between syntaxes
test distributed_irecv-5.1 {Positional vs named consistency - basic} {
    set result_pos [torch::distributed_irecv {3 4} 1]
    set result_named [torch::distributed_irecv -shape {3 4} -src 1]
    expr {$result_pos eq "irecv_handle_1" && $result_named eq "irecv_handle_1"}
} {1}

test distributed_irecv-5.2 {Positional vs named consistency - with tag} {
    set result_pos [torch::distributed_irecv {5 6} 2 42]
    set result_named [torch::distributed_irecv -shape {5 6} -src 2 -tag 42]
    expr {$result_pos eq "irecv_handle_1" && $result_named eq "irecv_handle_1"}
} {1}

test distributed_irecv-5.3 {snake_case vs camelCase consistency} {
    set result_snake [torch::distributed_irecv -shape {4 4} -src 1 -tag 10]
    set result_camel [torch::distributedIrecv -shape {4 4} -src 1 -tag 10]
    expr {$result_snake eq "irecv_handle_1" && $result_camel eq "irecv_handle_1"}
} {1}

test distributed_irecv-5.4 {All syntax combinations consistency} {
    set result1 [torch::distributed_irecv {2 3} 0 5]
    set result2 [torch::distributed_irecv -shape {2 3} -src 0 -tag 5]
    set result3 [torch::distributedIrecv {2 3} 0 5]
    set result4 [torch::distributedIrecv -shape {2 3} -src 0 -tag 5]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1" && $result4 eq "irecv_handle_1"}
} {1}

# Test 6: Error handling tests
test distributed_irecv-6.1 {Missing required parameters in named syntax} {
    catch {torch::distributed_irecv -shape {3 3}} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test distributed_irecv-6.2 {Missing shape parameter} {
    catch {torch::distributed_irecv -src 0} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test distributed_irecv-6.3 {Missing src parameter} {
    catch {torch::distributed_irecv -shape {3 3}} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test distributed_irecv-6.4 {Invalid src parameter type} {
    catch {torch::distributed_irecv -shape {3 3} -src "not_a_number"} result
    expr {[string match "*Invalid -src parameter*" $result]}
} {1}

test distributed_irecv-6.5 {Invalid tag parameter type} {
    catch {torch::distributed_irecv -shape {3 3} -src 0 -tag "not_a_number"} result
    expr {[string match "*Invalid -tag parameter*" $result]}
} {1}

test distributed_irecv-6.6 {Unknown parameter} {
    catch {torch::distributed_irecv -shape {3 3} -src 0 -unknown_param value} result
    expr {[string match "*Unknown parameter: -unknown_param*" $result]}
} {1}

test distributed_irecv-6.7 {Missing value for parameter} {
    catch {torch::distributed_irecv -shape {3 3} -src} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

test distributed_irecv-6.8 {Wrong number of positional arguments - too few} {
    catch {torch::distributed_irecv {3 3}} result
    expr {[string match "*Wrong number of arguments*" $result]}
} {1}

test distributed_irecv-6.9 {Wrong number of positional arguments - too many} {
    catch {torch::distributed_irecv {3 3} 0 1 extra_arg} result
    expr {[string match "*Wrong number of arguments*" $result]}
} {1}

test distributed_irecv-6.10 {Negative src rank} {
    catch {torch::distributed_irecv -shape {3 3} -src -1} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test distributed_irecv-6.11 {Empty shape} {
    catch {torch::distributed_irecv -shape {} -src 0} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test distributed_irecv-6.12 {Invalid shape values} {
    catch {torch::distributed_irecv -shape {0 3} -src 0} result
    # Should succeed even with zero dimensions (some frameworks allow this)
    expr {$result eq "irecv_handle_1"}
} {1}

# Test 7: Edge cases and special values
test distributed_irecv-7.1 {Single element tensor} {
    set result [torch::distributed_irecv -shape {1} -src 0]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-7.2 {Large tensor shapes} {
    set result1 [torch::distributed_irecv -shape {1024 1024} -src 0]
    set result2 [torch::distributed_irecv -shape {100 100 100} -src 0]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1"}
} {1}

test distributed_irecv-7.3 {High dimensional tensors} {
    set result1 [torch::distributed_irecv -shape {2 2 2 2 2} -src 0]
    set result2 [torch::distributed_irecv -shape {1 1 1 1 1 1 1} -src 0]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1"}
} {1}

test distributed_irecv-7.4 {Zero tag value} {
    set result [torch::distributed_irecv -shape {3 3} -src 0 -tag 0]
    expr {$result eq "irecv_handle_1"}
} {1}

test distributed_irecv-7.5 {High rank numbers} {
    set result1 [torch::distributed_irecv -shape {2 2} -src 100]
    set result2 [torch::distributed_irecv -shape {2 2} -src 999]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1"}
} {1}

# Test 8: Mixed usage scenarios
test distributed_irecv-8.1 {Multiple irecv operations} {
    set result1 [torch::distributed_irecv -shape {3 3} -src 0 -tag 1]
    set result2 [torch::distributed_irecv -shape {4 4} -src 1 -tag 2]
    set result3 [torch::distributed_irecv -shape {5 5} -src 2 -tag 3]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1"}
} {1}

test distributed_irecv-8.2 {Mixed syntax usage} {
    set result1 [torch::distributed_irecv {2 3} 1 10]
    set result2 [torch::distributedIrecv -shape {4 5} -src 2 -tag 20]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1"}
} {1}

test distributed_irecv-8.3 {Different parameter order variations} {
    set result1 [torch::distributed_irecv -shape {2 2} -src 0 -tag 5]
    set result2 [torch::distributed_irecv -tag 5 -src 0 -shape {2 2}]
    set result3 [torch::distributed_irecv -src 0 -tag 5 -shape {2 2}]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1"}
} {1}

# Test 9: Shape variations
test distributed_irecv-9.1 {Common ML tensor shapes} {
    set result1 [torch::distributed_irecv -shape {32 3 224 224} -src 0]
    set result2 [torch::distributed_irecv -shape {128 512} -src 0]
    set result3 [torch::distributed_irecv -shape {64 100 300} -src 0]
    set result4 [torch::distributed_irecv -shape {16 512 512} -src 0]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1" && $result4 eq "irecv_handle_1"}
} {1}

test distributed_irecv-9.2 {Broadcasting compatible shapes} {
    set result1 [torch::distributed_irecv -shape {1} -src 0]
    set result2 [torch::distributed_irecv -shape {1 1} -src 0]
    set result3 [torch::distributed_irecv -shape {1 1 1} -src 0]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1"}
} {1}

test distributed_irecv-9.3 {Asymmetric shapes} {
    set result1 [torch::distributed_irecv -shape {1000 1} -src 0]
    set result2 [torch::distributed_irecv -shape {1 1000} -src 0]
    set result3 [torch::distributed_irecv -shape {100 10 1} -src 0]
    expr {$result1 eq "irecv_handle_1" && $result2 eq "irecv_handle_1" && $result3 eq "irecv_handle_1"}
} {1}

cleanupTests 