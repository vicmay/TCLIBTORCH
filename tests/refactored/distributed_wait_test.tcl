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

# Helper function to check if result is a valid tensor handle
proc is_tensor_handle {result} {
    return [expr {[string match "tensor*" $result] || [string match "*handle*" $result]}]
}

# =============================================================================
# POSITIONAL SYNTAX TESTS (Backward Compatibility)
# =============================================================================

test distributed_wait-1.1 {Basic positional syntax with isend handle} {
    set result [torch::distributed_wait "isend_handle_1"]
    expr {$result eq "send_completed"}
} 1

test distributed_wait-1.2 {Basic positional syntax with irecv handle} {
    set result [torch::distributed_wait "irecv_handle_1"]
    # Should return a tensor handle
    expr {[is_tensor_handle $result]}
} 1

test distributed_wait-1.3 {Basic positional syntax with generic handle} {
    set result [torch::distributed_wait "generic_handle"]
    expr {$result eq "operation_completed"}
} 1

test distributed_wait-1.4 {Basic positional syntax with complex isend handle} {
    set result [torch::distributed_wait "isend_handle_dst2_tag5"]
    expr {$result eq "send_completed"}
} 1

test distributed_wait-1.5 {Basic positional syntax with complex irecv handle} {
    set result [torch::distributed_wait "irecv_handle_src1_tag3"]
    # Should return a tensor handle
    expr {[is_tensor_handle $result]}
} 1

# =============================================================================
# NAMED PARAMETER SYNTAX TESTS (Modern)
# =============================================================================

test distributed_wait-2.1 {Named parameter syntax with isend handle} {
    set result [torch::distributed_wait -handle "isend_handle_1"]
    expr {$result eq "send_completed"}
} 1

test distributed_wait-2.2 {Named parameter syntax with irecv handle} {
    set result [torch::distributed_wait -handle "irecv_handle_1"]
    # Should return a tensor handle
    expr {[is_tensor_handle $result]}
} 1

test distributed_wait-2.3 {Named parameter syntax with generic handle} {
    set result [torch::distributed_wait -handle "generic_handle"]
    expr {$result eq "operation_completed"}
} 1

test distributed_wait-2.4 {Named parameter syntax with complex isend handle} {
    set result [torch::distributed_wait -handle "isend_handle_dst2_tag5"]
    expr {$result eq "send_completed"}
} 1

test distributed_wait-2.5 {Named parameter syntax with complex irecv handle} {
    set result [torch::distributed_wait -handle "irecv_handle_src1_tag3"]
    # Should return a tensor handle
    expr {[is_tensor_handle $result]}
} 1

# =============================================================================
# CAMELCASE ALIAS TESTS
# =============================================================================

test distributed_wait-3.1 {CamelCase alias with isend handle} {
    set result [torch::distributedWait "isend_handle_1"]
    expr {$result eq "send_completed"}
} 1

test distributed_wait-3.2 {CamelCase alias with irecv handle} {
    set result [torch::distributedWait "irecv_handle_1"]
    # Should return a tensor handle
    expr {[is_tensor_handle $result]}
} 1

test distributed_wait-3.3 {CamelCase alias with named parameter syntax} {
    set result [torch::distributedWait -handle "isend_handle_1"]
    expr {$result eq "send_completed"}
} 1

test distributed_wait-3.4 {CamelCase alias with generic handle} {
    set result [torch::distributedWait -handle "generic_handle"]
    expr {$result eq "operation_completed"}
} 1

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

test distributed_wait-4.1 {Error: No arguments (positional)} {
    catch {torch::distributed_wait} msg
    expr {[string match "*Wrong number of arguments*" $msg] || [string match "*Required parameters missing*" $msg]}
} 1

test distributed_wait-4.2 {Error: Too many arguments (positional)} {
    catch {torch::distributed_wait "handle1" "handle2"} msg
    expr {[string match "*Wrong number of arguments*" $msg]}
} 1

test distributed_wait-4.3 {Error: Missing value for named parameter} {
    catch {torch::distributed_wait -handle} msg
    expr {[string match "*Missing value for parameter*" $msg]}
} 1

test distributed_wait-4.4 {Error: Unknown parameter} {
    catch {torch::distributed_wait -unknown_param "value"} msg
    expr {[string match "*Unknown parameter*" $msg]}
} 1

test distributed_wait-4.5 {Error: No parameters at all} {
    catch {torch::distributed_wait} msg
    expr {[string match "*Required parameters missing*" $msg] || [string match "*Wrong number of arguments*" $msg]}
} 1

test distributed_wait-4.6 {Error: Invalid parameter syntax} {
    catch {torch::distributed_wait -handle "value" -invalid} msg
    expr {[string match "*Missing value for parameter*" $msg]}
} 1

test distributed_wait-4.7 {Error: Multiple unknown parameters} {
    catch {torch::distributed_wait -handle "test" -unknown1 "val1" -unknown2 "val2"} msg
    expr {[string match "*Unknown parameter*" $msg]}
} 1

# =============================================================================
# EDGE CASE TESTS
# =============================================================================

test distributed_wait-5.1 {Edge case: Empty handle} {
    set result [torch::distributed_wait ""]
    expr {$result eq "operation_completed"}
} 1

test distributed_wait-5.2 {Edge case: Handle with mixed case} {
    set result [torch::distributed_wait "IsEnd_Handle_1"]
    expr {$result eq "operation_completed"}
} 1

test distributed_wait-5.3 {Edge case: Handle with mixed case irecv} {
    set result [torch::distributed_wait "IRecv_Handle_1"]
    # Mixed case "IRecv" doesn't match "irecv", so should return operation_completed
    expr {$result eq "operation_completed"}
} 1

test distributed_wait-5.4 {Edge case: Handle with special characters} {
    set result [torch::distributed_wait "handle-with_special.chars@123"]
    expr {$result eq "operation_completed"}
} 1

test distributed_wait-5.5 {Edge case: Handle with spaces} {
    set result [torch::distributed_wait "handle with spaces"]
    expr {$result eq "operation_completed"}
} 1

test distributed_wait-5.6 {Edge case: Handle starting with dash (positional)} {
    set result [torch::distributed_wait "-handle_like_param"]
    expr {$result eq "operation_completed"}
} 1

test distributed_wait-5.7 {Edge case: Very long handle name} {
    set long_handle [string repeat "a" 1000]
    set result [torch::distributed_wait $long_handle]
    expr {$result eq "operation_completed"}
} 1

# =============================================================================
# HANDLE TYPE TESTS
# =============================================================================

test distributed_wait-6.1 {Handle type: isend variations} {
    set handles [list "isend_handle_1" "test_isend_handle" "isend_dst1_tag2"]
    set all_correct 1
    foreach handle $handles {
        set result [torch::distributed_wait $handle]
        if {$result ne "send_completed"} {
            set all_correct 0
            break
        }
    }
    expr {$all_correct}
} 1

test distributed_wait-6.2 {Handle type: irecv variations} {
    set handles [list "irecv_handle_1" "test_irecv_handle" "irecv_src1_tag2"]
    set all_correct 1
    foreach handle $handles {
        set result [torch::distributed_wait $handle]
        if {![is_tensor_handle $result]} {
            set all_correct 0
            break
        }
    }
    expr {$all_correct}
} 1

test distributed_wait-6.3 {Handle type: generic variations} {
    set handles [list "generic_handle" "test_handle" "operation_123"]
    set all_correct 1
    foreach handle $handles {
        set result [torch::distributed_wait $handle]
        if {$result ne "operation_completed"} {
            set all_correct 0
            break
        }
    }
    expr {$all_correct}
} 1

# =============================================================================
# CONSISTENCY TESTS
# =============================================================================

test distributed_wait-7.1 {Consistency: Positional and named syntax produce same result (isend)} {
    set result1 [torch::distributed_wait "isend_test_handle"]
    set result2 [torch::distributed_wait -handle "isend_test_handle"]
    expr {$result1 eq $result2}
} 1

test distributed_wait-7.2 {Consistency: Positional and named syntax produce same result (irecv)} {
    set result1 [torch::distributed_wait "irecv_test_handle"]
    set result2 [torch::distributed_wait -handle "irecv_test_handle"]
    # Both should return tensor handles
    expr {[is_tensor_handle $result1] && [is_tensor_handle $result2]}
} 1

test distributed_wait-7.3 {Consistency: Original and camelCase produce same result} {
    set result1 [torch::distributed_wait "test_handle"]
    set result2 [torch::distributedWait "test_handle"]
    expr {$result1 eq $result2}
} 1

test distributed_wait-7.4 {Consistency: All syntax variations produce same result} {
    set result1 [torch::distributed_wait "isend_test_handle"]
    set result2 [torch::distributed_wait -handle "isend_test_handle"]
    set result3 [torch::distributedWait "isend_test_handle"]
    set result4 [torch::distributedWait -handle "isend_test_handle"]
    expr {$result1 eq $result2 && $result2 eq $result3 && $result3 eq $result4}
} 1

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

test distributed_wait-8.1 {Integration: Wait for isend operation} {
    set handle "isend_handle_dst0_tag0"
    set result [torch::distributed_wait $handle]
    expr {$result eq "send_completed"}
} 1

test distributed_wait-8.2 {Integration: Wait for irecv operation} {
    set handle "irecv_handle_src1_tag5"
    set result [torch::distributed_wait $handle]
    # Should return a tensor handle
    expr {[is_tensor_handle $result]}
} 1

test distributed_wait-8.3 {Integration: Wait for multiple operations} {
    set handles [list "isend_handle_1" "irecv_handle_1" "generic_handle"]
    set results [list]
    foreach handle $handles {
        lappend results [torch::distributed_wait $handle]
    }
    # Check if we got expected results
    set result1 [lindex $results 0]
    set result2 [lindex $results 1]
    set result3 [lindex $results 2]
    expr {$result1 eq "send_completed" && [is_tensor_handle $result2] && $result3 eq "operation_completed"}
} 1

# =============================================================================
# PERFORMANCE AND STRESS TESTS
# =============================================================================

test distributed_wait-9.1 {Performance: Many consecutive isend waits} {
    set success 1
    for {set i 0} {$i < 50} {incr i} {
        set result [torch::distributed_wait "isend_handle_$i"]
        if {$result ne "send_completed"} {
            set success 0
            break
        }
    }
    expr {$success}
} 1

test distributed_wait-9.2 {Performance: Many consecutive irecv waits} {
    set success 1
    for {set i 0} {$i < 50} {incr i} {
        set result [torch::distributed_wait "irecv_handle_$i"]
        if {![is_tensor_handle $result]} {
            set success 0
            break
        }
    }
    expr {$success}
} 1

test distributed_wait-9.3 {Performance: Alternating syntax calls} {
    set success 1
    for {set i 0} {$i < 25} {incr i} {
        set result1 [torch::distributed_wait "isend_handle_$i"]
        set result2 [torch::distributed_wait -handle "isend_handle_$i"]
        if {$result1 ne "send_completed" || $result2 ne "send_completed"} {
            set success 0
            break
        }
    }
    expr {$success}
} 1

# =============================================================================
# MATHEMATICAL CORRECTNESS TESTS
# =============================================================================

test distributed_wait-10.1 {Mathematical correctness: irecv returns valid tensor} {
    set result [torch::distributed_wait "irecv_handle_1"]
    # For irecv operations, we expect a tensor handle
    expr {[is_tensor_handle $result]}
} 1

test distributed_wait-10.2 {Mathematical correctness: Multiple irecv operations} {
    set handles [list "irecv_handle_1" "irecv_handle_2" "irecv_handle_3"]
    set all_tensors 1
    foreach handle $handles {
        set result [torch::distributed_wait $handle]
        if {![is_tensor_handle $result]} {
            set all_tensors 0
            break
        }
    }
    expr {$all_tensors}
} 1

# =============================================================================
# FINAL VALIDATION TEST
# =============================================================================

test distributed_wait-11.1 {Final validation: All syntax forms work correctly} {
    set all_good 1
    
    # Test all major syntax forms with different handle types
    set test_cases [list \
        {"torch::distributed_wait isend_handle_1" "send_completed"} \
        {"torch::distributed_wait -handle isend_handle_1" "send_completed"} \
        {"torch::distributedWait isend_handle_1" "send_completed"} \
        {"torch::distributedWait -handle isend_handle_1" "send_completed"} \
        {"torch::distributed_wait generic_handle" "operation_completed"} \
        {"torch::distributed_wait -handle generic_handle" "operation_completed"} \
    ]
    
    foreach test_case $test_cases {
        set cmd [lindex $test_case 0]
        set expected [lindex $test_case 1]
        
        set result [eval $cmd]
        if {$result ne $expected} {
            set all_good 0
            break
        }
    }
    
    # Test irecv handles separately since they return tensor handles
    set irecv_tests [list \
        "torch::distributed_wait irecv_handle_1" \
        "torch::distributed_wait -handle irecv_handle_1" \
        "torch::distributedWait irecv_handle_1" \
        "torch::distributedWait -handle irecv_handle_1" \
    ]
    
    foreach irecv_test $irecv_tests {
        set result [eval $irecv_test]
        if {![is_tensor_handle $result]} {
            set all_good 0
            break
        }
    }
    
    expr {$all_good}
} 1

puts "distributed_wait: All tests completed"
cleanupTests 