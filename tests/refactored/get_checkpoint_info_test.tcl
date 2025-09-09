#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# ----------------------------------------------------------------------------
# get_checkpoint_info tests
# ----------------------------------------------------------------------------

# Create a dummy checkpoint using save_checkpoint helper if available
# We will skip creation if command not found, just test error handling

if {[catch {info command torch::save_checkpoint}] == 0} {
    # Create model & optimizer placeholders (assumed helpers exist)
    # For the purpose of this unit test we only verify call path; if helpers
    # are missing we will skip functional tests.
    set skipFunctional 0
} else {
    set skipFunctional 1
}

# Positional syntax should error on missing file

test gci-1.1 {Error on missing argument} {
    catch {torch::get_checkpoint_info} msg
    expr {[string match "*wrong # args*" $msg] || [string match "*args*" $msg]}
} {1}

# Named syntax should error on unknown parameter

test gci-1.2 {Error on unknown parameter} {
    catch {torch::getCheckpointInfo -foo bar} msg
    string match "*Unknown parameter*" $msg
} {1}

# Named syntax missing value

test gci-1.3 {Error on missing value for -file} {
    catch {torch::getCheckpointInfo -file} msg
    string match "*Missing value for option*" $msg
} {1}

# Skip remaining functional tests if helpers unavailable
if {$skipFunctional} {
    puts "Skipping functional tests for get_checkpoint_info (save_checkpoint not available)"
    cleanupTests
    exit 0
}

# TODO: Add functional tests once checkpoint helpers are guaranteed available

cleanupTests 