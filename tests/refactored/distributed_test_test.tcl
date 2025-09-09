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

# =============================================================================
# POSITIONAL SYNTAX TESTS (Backward Compatibility)
# =============================================================================

test distributed_test-1.1 {Basic positional syntax with isend handle} {
    set result [torch::distributed_test "isend_handle_1"]
    expr {$result eq "true"}
} 1

test distributed_test-1.2 {Basic positional syntax with irecv handle} {
    set result [torch::distributed_test "irecv_handle_1"]
    expr {$result eq "true"}
} 1

test distributed_test-1.3 {Basic positional syntax with custom handle} {
    set result [torch::distributed_test "custom_handle_test"]
    expr {$result eq "true"}
} 1

test distributed_test-1.4 {Basic positional syntax with empty handle} {
    set result [torch::distributed_test ""]
    expr {$result eq "true"}
} 1

test distributed_test-1.5 {Basic positional syntax with complex handle} {
    set result [torch::distributed_test "isend_handle_dst2_tag5"]
    expr {$result eq "true"}
} 1

# =============================================================================
# NAMED PARAMETER SYNTAX TESTS (Modern)
# =============================================================================

test distributed_test-2.1 {Named parameter syntax with isend handle} {
    set result [torch::distributed_test -handle "isend_handle_1"]
    expr {$result eq "true"}
} 1

test distributed_test-2.2 {Named parameter syntax with irecv handle} {
    set result [torch::distributed_test -handle "irecv_handle_1"]
    expr {$result eq "true"}
} 1

test distributed_test-2.3 {Named parameter syntax with custom handle} {
    set result [torch::distributed_test -handle "custom_handle_test"]
    expr {$result eq "true"}
} 1

test distributed_test-2.4 {Named parameter syntax with empty handle} {
    set result [torch::distributed_test -handle ""]
    expr {$result eq "true"}
} 1

test distributed_test-2.5 {Named parameter syntax with complex handle} {
    set result [torch::distributed_test -handle "isend_handle_dst2_tag5"]
    expr {$result eq "true"}
} 1

# =============================================================================
# CAMELCASE ALIAS TESTS
# =============================================================================

test distributed_test-3.1 {CamelCase alias with positional syntax} {
    set result [torch::distributedTest "isend_handle_1"]
    expr {$result eq "true"}
} 1

test distributed_test-3.2 {CamelCase alias with named parameter syntax} {
    set result [torch::distributedTest -handle "irecv_handle_1"]
    expr {$result eq "true"}
} 1

test distributed_test-3.3 {CamelCase alias with custom handle} {
    set result [torch::distributedTest -handle "custom_handle_test"]
    expr {$result eq "true"}
} 1

test distributed_test-3.4 {CamelCase alias with empty handle} {
    set result [torch::distributedTest -handle ""]
    expr {$result eq "true"}
} 1

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

test distributed_test-4.1 {Error: No arguments (positional)} {
    catch {torch::distributed_test} msg
    expr {[string match "*Wrong number of arguments*" $msg] || [string match "*Required parameters missing*" $msg]}
} 1

test distributed_test-4.2 {Error: Too many arguments (positional)} {
    catch {torch::distributed_test "handle1" "handle2"} msg
    expr {[string match "*Wrong number of arguments*" $msg]}
} 1

test distributed_test-4.3 {Error: Missing value for named parameter} {
    catch {torch::distributed_test -handle} msg
    expr {[string match "*Missing value for parameter*" $msg]}
} 1

test distributed_test-4.4 {Error: Unknown parameter} {
    catch {torch::distributed_test -unknown_param "value"} msg
    expr {[string match "*Unknown parameter*" $msg]}
} 1

test distributed_test-4.5 {Error: No parameters at all} {
    catch {torch::distributed_test} msg
    expr {[string match "*Required parameters missing*" $msg] || [string match "*Wrong number of arguments*" $msg]}
} 1

test distributed_test-4.6 {Error: Invalid parameter syntax} {
    catch {torch::distributed_test -handle "value" -invalid} msg
    expr {[string match "*Missing value for parameter*" $msg]}
} 1

test distributed_test-4.7 {Error: Multiple unknown parameters} {
    catch {torch::distributed_test -handle "test" -unknown1 "val1" -unknown2 "val2"} msg
    expr {[string match "*Unknown parameter*" $msg]}
} 1

# =============================================================================
# EDGE CASE TESTS
# =============================================================================

test distributed_test-5.1 {Edge case: Very long handle name} {
    set long_handle [string repeat "a" 1000]
    set result [torch::distributed_test $long_handle]
    expr {$result eq "true"}
} 1

test distributed_test-5.2 {Edge case: Handle with special characters} {
    set result [torch::distributed_test "handle-with_special.chars@123"]
    expr {$result eq "true"}
} 1

test distributed_test-5.3 {Edge case: Handle with spaces} {
    set result [torch::distributed_test "handle with spaces"]
    expr {$result eq "true"}
} 1

test distributed_test-5.4 {Edge case: Handle with numbers only} {
    set result [torch::distributed_test "12345"]
    expr {$result eq "true"}
} 1

test distributed_test-5.5 {Edge case: Handle with unicode characters} {
    set result [torch::distributed_test "handle_üñíçødé"]
    expr {$result eq "true"}
} 1

test distributed_test-5.6 {Edge case: Handle starting with dash (named param with positional)} {
    set result [torch::distributed_test "-handle_like_param"]
    expr {$result eq "true"}
} 1

# =============================================================================
# CONSISTENCY TESTS
# =============================================================================

test distributed_test-6.1 {Consistency: Positional and named syntax produce same result} {
    set result1 [torch::distributed_test "test_handle"]
    set result2 [torch::distributed_test -handle "test_handle"]
    expr {$result1 eq $result2}
} 1

test distributed_test-6.2 {Consistency: Original and camelCase produce same result} {
    set result1 [torch::distributed_test "test_handle"]
    set result2 [torch::distributedTest "test_handle"]
    expr {$result1 eq $result2}
} 1

test distributed_test-6.3 {Consistency: All syntax variations produce same result} {
    set result1 [torch::distributed_test "test_handle"]
    set result2 [torch::distributed_test -handle "test_handle"]
    set result3 [torch::distributedTest "test_handle"]
    set result4 [torch::distributedTest -handle "test_handle"]
    expr {$result1 eq $result2 && $result2 eq $result3 && $result3 eq $result4}
} 1

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

test distributed_test-7.1 {Integration: Test with isend-like handle} {
    set handle "isend_handle_dst0_tag0"
    set result [torch::distributed_test $handle]
    expr {$result eq "true"}
} 1

test distributed_test-7.2 {Integration: Test with irecv-like handle} {
    set handle "irecv_handle_src1_tag5"
    set result [torch::distributed_test $handle]
    expr {$result eq "true"}
} 1

test distributed_test-7.3 {Integration: Test multiple handles sequentially} {
    set handles [list "handle1" "handle2" "handle3"]
    set all_true 1
    foreach handle $handles {
        set result [torch::distributed_test $handle]
        if {$result ne "true"} {
            set all_true 0
            break
        }
    }
    expr {$all_true}
} 1

# =============================================================================
# PERFORMANCE AND STRESS TESTS
# =============================================================================

test distributed_test-8.1 {Performance: Many consecutive calls} {
    set success 1
    for {set i 0} {$i < 100} {incr i} {
        set result [torch::distributed_test "handle_$i"]
        if {$result ne "true"} {
            set success 0
            break
        }
    }
    expr {$success}
} 1

test distributed_test-8.2 {Performance: Alternating syntax calls} {
    set success 1
    for {set i 0} {$i < 50} {incr i} {
        set result1 [torch::distributed_test "handle_$i"]
        set result2 [torch::distributed_test -handle "handle_$i"]
        if {$result1 ne "true" || $result2 ne "true"} {
            set success 0
            break
        }
    }
    expr {$success}
} 1

# =============================================================================
# FINAL VALIDATION TEST
# =============================================================================

test distributed_test-9.1 {Final validation: All syntax forms work correctly} {
    set all_good 1
    
    # Test all major syntax forms
    set forms [list \
        {torch::distributed_test "test_handle"} \
        {torch::distributed_test -handle "test_handle"} \
        {torch::distributedTest "test_handle"} \
        {torch::distributedTest -handle "test_handle"} \
    ]
    
    foreach form $forms {
        set result [eval $form]
        if {$result ne "true"} {
            set all_good 0
            break
        }
    }
    
    expr {$all_good}
} 1

puts "distributed_test: All tests completed"
cleanupTests 