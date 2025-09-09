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
# Tests for torch::lr_scheduler_multi_step - Multi-step Learning Rate Scheduler
# ============================================================================

# Since we don't have real optimizers for testing, we'll focus on parameter
# parsing and error handling. Most tests will expect "Invalid optimizer name"
# error, but the important thing is that the parsing works correctly.

# ============================================================================
# Tests for Positional Syntax (Backward Compatibility)
# ============================================================================

test lr_scheduler_multi_step-1.1 {Positional syntax - parameter parsing works} {
    # Test that positional syntax is parsed correctly (invalid optimizer is expected)
    catch {torch::lr_scheduler_multi_step test_opt {10 20 30}} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multi_step-1.2 {Positional syntax with gamma - parameter parsing works} {
    catch {torch::lr_scheduler_multi_step test_opt {5 15 25} 0.5} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multi_step-1.3 {Positional syntax - wrong number of args} {
    catch {torch::lr_scheduler_multi_step test_opt} result
    string match "*Usage: torch::lr_scheduler_multi_step*" $result
} {1}

test lr_scheduler_multi_step-1.4 {Positional syntax - too many args} {
    catch {torch::lr_scheduler_multi_step test_opt {10} 0.5 extra} result
    string match "*Usage: torch::lr_scheduler_multi_step*" $result
} {1}

# ============================================================================
# Tests for Named Parameter Syntax
# ============================================================================

test lr_scheduler_multi_step-2.1 {Named parameter syntax - basic parsing} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones {10 20 30}} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multi_step-2.2 {Named parameter syntax - with gamma} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones {5 15} -gamma 0.3} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multi_step-2.3 {Named parameter syntax - different order} {
    catch {torch::lr_scheduler_multi_step -milestones {8 16} -gamma 0.7 -optimizer test_opt} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multi_step-2.4 {Named parameter syntax - missing optimizer} {
    catch {torch::lr_scheduler_multi_step -milestones {10 20}} result
    string match "*Required parameters missing*" $result
} {1}

test lr_scheduler_multi_step-2.5 {Named parameter syntax - missing milestones} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt} result
    string match "*Required parameters missing*" $result
} {1}

# ============================================================================
# Tests for camelCase Alias
# ============================================================================

test lr_scheduler_multi_step-3.1 {camelCase alias - positional syntax} {
    catch {torch::lrSchedulerMultiStep test_opt {10 20}} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multi_step-3.2 {camelCase alias - named parameters} {
    catch {torch::lrSchedulerMultiStep -optimizer test_opt -milestones {15 30} -gamma 0.4} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multi_step-3.3 {camelCase alias exists} {
    # Just check that the command exists
    expr {[llength [info commands torch::lrSchedulerMultiStep]] > 0}
} {1}

# ============================================================================
# Tests for Error Handling
# ============================================================================

test lr_scheduler_multi_step-4.1 {Error handling - unknown parameter} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones {10} -invalid param} result
    string match "*Unknown parameter: -invalid*" $result
} {1}

test lr_scheduler_multi_step-4.2 {Error handling - invalid gamma value} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones {10} -gamma invalid} result
    string match "*Invalid gamma value*" $result
} {1}

test lr_scheduler_multi_step-4.3 {Error handling - empty milestones} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones {}} result
    string match "*Required parameters missing*" $result
} {1}

test lr_scheduler_multi_step-4.4 {Error handling - invalid milestone value} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones {10 invalid 20}} result
    string match "*Invalid milestone value*" $result
} {1}

test lr_scheduler_multi_step-4.5 {Error handling - missing value for parameter} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones} result
    string match "*Missing value for parameter*" $result
} {1}

test lr_scheduler_multi_step-4.6 {Error handling - invalid milestones list} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones "\{invalid_list"} result
    string match "*Invalid milestones list*" $result
} {1}

# ============================================================================
# Tests for Parameter Validation
# ============================================================================

test lr_scheduler_multi_step-5.1 {Parameter validation - positive gamma values} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones {10} -gamma 0.5} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multi_step-5.2 {Parameter validation - gamma = 1.0} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones {10} -gamma 1.0} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multi_step-5.3 {Parameter validation - negative gamma (should work)} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones {10} -gamma -0.5} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multi_step-5.4 {Parameter validation - zero gamma} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones {10} -gamma 0.0} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multi_step-5.5 {Parameter validation - single milestone} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones {5}} result
    set result
} {Invalid optimizer name}

test lr_scheduler_multi_step-5.6 {Parameter validation - multiple milestones} {
    catch {torch::lr_scheduler_multi_step -optimizer test_opt -milestones {1 5 10 20 50}} result
    set result
} {Invalid optimizer name}

# ============================================================================
# Command Existence Tests
# ============================================================================

test lr_scheduler_multi_step-6.1 {Command exists - snake_case} {
    expr {[llength [info commands torch::lr_scheduler_multi_step]] > 0}
} {1}

test lr_scheduler_multi_step-6.2 {Command exists - camelCase alias} {
    expr {[llength [info commands torch::lrSchedulerMultiStep]] > 0}
} {1}

cleanupTests 