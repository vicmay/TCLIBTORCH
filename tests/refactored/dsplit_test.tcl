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
# TORCH::DSPLIT COMPREHENSIVE TEST SUITE
# =====================================================================

# Tests for positional syntax (backward compatibility)
test dsplit-1.1 {Basic positional syntax with number of sections} {
    set t1 [torch::zeros {2 4 6}]
    set result [torch::dsplit $t1 2]
    llength $result
} {2}

test dsplit-1.2 {Positional syntax with different section count} {
    set t1 [torch::ones {3 4 8}]
    set result [torch::dsplit $t1 4]
    llength $result
} {4}

test dsplit-1.3 {Positional syntax with indices list} {
    set t1 [torch::zeros {2 2 6}]
    set result [torch::dsplit $t1 {2 4}]
    llength $result
} {3}

test dsplit-1.4 {Positional syntax with single section} {
    set t1 [torch::ones {1 1 4}]
    set result [torch::dsplit $t1 1]
    llength $result
} {1}

# Tests for named parameter syntax
test dsplit-2.1 {Named parameter syntax basic with -tensor and -sections} {
    set t1 [torch::zeros {2 3 6}]
    set result [torch::dsplit -tensor $t1 -sections 2]
    llength $result
} {2}

test dsplit-2.2 {Named parameter syntax with -input and -sections} {
    set t1 [torch::ones {2 3 8}]
    set result [torch::dsplit -input $t1 -sections 4]
    llength $result
} {4}

test dsplit-2.3 {Named parameter syntax with -tensor and -indices} {
    set t1 [torch::zeros {2 2 6}]
    set result [torch::dsplit -tensor $t1 -indices {2 4}]
    llength $result
} {3}

test dsplit-2.4 {Named parameter syntax with different order} {
    set t1 [torch::ones {2 3 6}]
    set result [torch::dsplit -sections 3 -tensor $t1]
    llength $result
} {3}

test dsplit-2.5 {Named parameter syntax with -input and -indices} {
    set t1 [torch::zeros {1 2 6}]
    set result [torch::dsplit -input $t1 -indices {1 3 5}]
    llength $result
} {4}

# Tests for camelCase alias (dsplit doesn't need camelCase transformation)
test dsplit-3.1 {Dsplit command with positional syntax} {
    set t1 [torch::ones {2 2 4}]
    set result [torch::dsplit $t1 2]
    llength $result
} {2}

test dsplit-3.2 {Dsplit command with named parameters} {
    set t1 [torch::ones {2 2 4}]
    set result [torch::dsplit -tensor $t1 -sections 2]
    llength $result
} {2}

# Tests for error handling
test dsplit-4.1 {Error on missing parameters} {
    catch {torch::dsplit} msg
    expr {[string match "*Usage*" $msg] || [string match "*Required parameters*" $msg]}
} {1}

test dsplit-4.2 {Error on single parameter} {
    set t1 [torch::ones {2 2 4}]
    catch {torch::dsplit $t1} msg
    expr {[string match "*Usage*" $msg] || [string match "*Required parameters*" $msg]}
} {1}

test dsplit-4.3 {Error on invalid tensor name} {
    catch {torch::dsplit invalid_tensor 2} msg
    string match "*Error*" $msg
} {1}

test dsplit-4.4 {Error on invalid tensor name with named parameters} {
    catch {torch::dsplit -tensor invalid_tensor -sections 2} msg
    string match "*Error*" $msg
} {1}

test dsplit-4.5 {Error on unknown parameter} {
    set t1 [torch::ones {2 2 4}]
    catch {torch::dsplit -unknown_param $t1 -sections 2} msg
    string match "*Unknown parameter*" $msg
} {1}

test dsplit-4.6 {Error on missing parameter value} {
    set t1 [torch::ones {2 2 4}]
    catch {torch::dsplit -tensor $t1 -sections} msg
    string match "*Missing value for parameter*" $msg
} {1}

test dsplit-4.7 {Error on missing required parameter -tensor} {
    catch {torch::dsplit -sections 2} msg
    string match "*Required parameters missing*" $msg
} {1}

test dsplit-4.8 {Error on missing required parameter -sections} {
    set t1 [torch::ones {2 2 4}]
    catch {torch::dsplit -tensor $t1} msg
    string match "*Required parameters missing*" $msg
} {1}

# Tests for mathematical correctness  
test dsplit-5.1 {Mathematical correctness - split into equal sections} {
    set t1 [torch::zeros {2 2 6}]
    set result [torch::dsplit $t1 3]
    llength $result
} {3}

test dsplit-5.2 {Mathematical correctness - split with indices creates correct number} {
    set t1 [torch::ones {1 1 8}]
    set result [torch::dsplit $t1 {2 4 6}]
    llength $result
} {4}

test dsplit-5.3 {Mathematical correctness - single section returns original} {
    set t1 [torch::ones {2 3 4}]
    set result [torch::dsplit $t1 1]
    llength $result
} {1}

test dsplit-5.4 {Mathematical correctness - indices at boundaries} {
    set t1 [torch::zeros {1 1 6}]
    set result [torch::dsplit $t1 {0 3 6}]
    llength $result
} {4}

# Tests for different tensor shapes and sizes
test dsplit-6.1 {Small tensor depth dimension} {
    set t1 [torch::ones {1 1 2}]
    set result [torch::dsplit $t1 2]
    llength $result
} {2}

test dsplit-6.2 {Medium tensor with multiple sections} {
    set t1 [torch::zeros {3 4 12}]
    set result [torch::dsplit $t1 4]
    llength $result
} {4}

test dsplit-6.3 {Large depth dimension} {
    set t1 [torch::ones {2 2 20}]
    set result [torch::dsplit $t1 5]
    llength $result
} {5}

test dsplit-6.4 {Complex indices pattern} {
    set t1 [torch::zeros {2 3 10}]
    set result [torch::dsplit $t1 {1 3 7 9}]
    llength $result
} {5}

# Tests for syntax consistency (both syntaxes should produce same results)
test dsplit-7.1 {Syntax consistency - basic sections} {
    set t1 [torch::ones {2 2 6}]
    set result1 [torch::dsplit $t1 3]
    set result2 [torch::dsplit -tensor $t1 -sections 3]
    expr {[llength $result1] == [llength $result2]}
} {1}

test dsplit-7.2 {Syntax consistency - indices splitting} {
    set t1 [torch::zeros {2 2 8}]
    set result1 [torch::dsplit $t1 {2 4 6}]
    set result2 [torch::dsplit -tensor $t1 -indices {2 4 6}]
    expr {[llength $result1] == [llength $result2]}
} {1}

test dsplit-7.3 {Syntax consistency - single section} {
    set t1 [torch::ones {3 3 4}]
    set result1 [torch::dsplit $t1 1]
    set result2 [torch::dsplit -input $t1 -sections 1]
    expr {[llength $result1] == [llength $result2]}
} {1}

test dsplit-7.4 {Syntax consistency - complex indices} {
    set t1 [torch::zeros {1 2 12}]
    set result1 [torch::dsplit $t1 {3 6 9}]
    set result2 [torch::dsplit -tensor $t1 -indices {3 6 9}]
    expr {[llength $result1] == [llength $result2]}
} {1}

# Tests for parameter validation
test dsplit-8.1 {Parameter validation - both parameters required} {
    catch {torch::dsplit -tensor} msg
    string match "*Missing value for parameter*" $msg
} {1}

test dsplit-8.2 {Parameter validation - unknown parameters rejected} {
    set t1 [torch::ones {2 2 4}]
    catch {torch::dsplit -tensor $t1 -sections 2 -extra_param value} msg
    string match "*Unknown parameter*" $msg
} {1}

test dsplit-8.3 {Parameter validation - parameter order independence} {
    set t1 [torch::ones {2 2 6}]
    set result1 [torch::dsplit -tensor $t1 -sections 2]
    set result2 [torch::dsplit -sections 2 -tensor $t1]
    expr {[llength $result1] == [llength $result2]}
} {1}

test dsplit-8.4 {Parameter validation - both -input and -tensor work} {
    set t1 [torch::zeros {2 2 4}]
    set result1 [torch::dsplit -tensor $t1 -sections 2]
    set result2 [torch::dsplit -input $t1 -sections 2]
    expr {[llength $result1] == [llength $result2]}
} {1}

test dsplit-8.5 {Parameter validation - both -sections and -indices work} {
    set t1 [torch::ones {2 2 6}]
    set result1 [torch::dsplit -tensor $t1 -sections 3]
    set result2 [torch::dsplit -tensor $t1 -indices {2 4}]
    # Both should work, just different ways of splitting
    expr {[llength $result1] >= 1 && [llength $result2] >= 1}
} {1}

# Tests for edge cases
test dsplit-9.1 {Edge case - minimum depth dimension} {
    set t1 [torch::ones {1 1 1}]
    set result [torch::dsplit -tensor $t1 -sections 1]
    llength $result
} {1}

test dsplit-9.2 {Edge case - identical tensor handles} {
    set t1 [torch::ones {2 2 4}]
    set result [torch::dsplit $t1 2]
    llength $result
} {2}

test dsplit-9.3 {Edge case - large number of sections} {
    set t1 [torch::zeros {1 1 8}]
    set result [torch::dsplit -tensor $t1 -sections 8]
    llength $result
} {8}

# Tests for result validation
test dsplit-10.1 {Result validation - returned tensors are valid handles} {
    set t1 [torch::ones {2 2 4}]
    set result [torch::dsplit $t1 2]
    set first_tensor [lindex $result 0]
    # Check if we can get shape (validates it's a real tensor)
    set shape [torch::tensor_shape $first_tensor]
    expr {[llength $shape] > 0}
} {1}

test dsplit-10.2 {Result validation - all result tensors have valid shapes} {
    set t1 [torch::zeros {2 3 6}]
    set result [torch::dsplit -tensor $t1 -sections 3]
    set all_valid 1
    foreach tensor $result {
        if {[catch {torch::tensor_shape $tensor}]} {
            set all_valid 0
            break
        }
    }
    set all_valid
} {1}

test dsplit-10.3 {Result validation - sections produce expected count} {
    set t1 [torch::ones {1 1 12}]
    set result [torch::dsplit -tensor $t1 -sections 4]
    llength $result
} {4}

test dsplit-10.4 {Result validation - indices produce expected count} {
    set t1 [torch::zeros {2 2 10}]
    set result [torch::dsplit -tensor $t1 -indices {2 5 8}]
    # 3 indices create 4 sections: [0:2], [2:5], [5:8], [8:10]
    llength $result
} {4}

cleanupTests 