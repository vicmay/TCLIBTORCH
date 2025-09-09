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

# Test group_norm with positional syntax
test group_norm-1.1 {Basic positional syntax} {
    set layer [torch::group_norm 2 4]
    set layer_id [string length $layer]
    expr {$layer_id > 0}
} {1}

test group_norm-1.2 {Positional syntax with eps} {
    set layer [torch::group_norm 2 4 1e-6]
    set layer_id [string length $layer]
    expr {$layer_id > 0}
} {1}

test group_norm-1.3 {Positional syntax with different groups} {
    set layer [torch::group_norm 4 8]
    set layer_id [string length $layer]
    expr {$layer_id > 0}
} {1}

# Test group_norm with named parameter syntax
test group_norm-2.1 {Named parameter syntax - basic} {
    set layer [torch::group_norm -num_groups 2 -num_channels 4]
    set layer_id [string length $layer]
    expr {$layer_id > 0}
} {1}

test group_norm-2.2 {Named parameter syntax - with eps} {
    set layer [torch::group_norm -num_groups 2 -num_channels 4 -eps 1e-6]
    set layer_id [string length $layer]
    expr {$layer_id > 0}
} {1}

test group_norm-2.3 {Named parameter syntax - camelCase parameters} {
    set layer [torch::group_norm -numGroups 2 -numChannels 4]
    set layer_id [string length $layer]
    expr {$layer_id > 0}
} {1}

test group_norm-2.4 {Named parameter syntax - mixed case} {
    set layer [torch::group_norm -numGroups 2 -num_channels 4 -eps 1e-6]
    set layer_id [string length $layer]
    expr {$layer_id > 0}
} {1}

# Test camelCase alias
test group_norm-3.1 {camelCase alias - basic} {
    set layer [torch::groupNorm -num_groups 2 -num_channels 4]
    set layer_id [string length $layer]
    expr {$layer_id > 0}
} {1}

test group_norm-3.2 {camelCase alias - with eps} {
    set layer [torch::groupNorm -numGroups 2 -numChannels 4 -eps 1e-6]
    set layer_id [string length $layer]
    expr {$layer_id > 0}
} {1}

test group_norm-3.3 {camelCase alias - positional syntax} {
    set layer [torch::groupNorm 4 8 1e-5]
    set layer_id [string length $layer]
    expr {$layer_id > 0}
} {1}

# Test layer functionality
test group_norm-4.1 {Layer functionality - basic creation} {
    set layer [torch::group_norm 2 4]
    set layer_valid [expr {[string match "groupnorm*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test group_norm-4.2 {Layer functionality - different configurations} {
    set layer [torch::group_norm 4 8]
    set layer_valid [expr {[string match "groupnorm*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test group_norm-4.3 {Layer functionality - with custom eps} {
    set layer [torch::group_norm 2 4 1e-6]
    set layer_valid [expr {[string match "groupnorm*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

# Test syntax consistency
test group_norm-5.1 {Syntax consistency - identical results} {
    set layer1 [torch::group_norm 2 4 1e-6]
    set layer2 [torch::group_norm -num_groups 2 -num_channels 4 -eps 1e-6]
    
    # Both should succeed
    set success1 [expr {[string length $layer1] > 0}]
    set success2 [expr {[string length $layer2] > 0}]
    
    expr {$success1 && $success2}
} {1}

test group_norm-5.2 {Syntax consistency - camelCase alias} {
    set layer1 [torch::group_norm -num_groups 2 -num_channels 4]
    set layer2 [torch::groupNorm -num_groups 2 -num_channels 4]
    
    # Both should succeed
    set success1 [expr {[string length $layer1] > 0}]
    set success2 [expr {[string length $layer2] > 0}]
    
    expr {$success1 && $success2}
} {1}

test group_norm-5.3 {Syntax consistency - parameter alternatives} {
    set layer1 [torch::group_norm -numGroups 2 -numChannels 4]
    set layer2 [torch::group_norm -num_groups 2 -num_channels 4]
    
    # Both should succeed
    set success1 [expr {[string length $layer1] > 0}]
    set success2 [expr {[string length $layer2] > 0}]
    
    expr {$success1 && $success2}
} {1}

# Test error handling
test group_norm-6.1 {Error handling - no parameters} {
    catch {torch::group_norm} msg
    expr {[string match "*numGroups*" $msg] || [string match "*numChannels*" $msg]}
} {1}

test group_norm-6.2 {Error handling - insufficient parameters} {
    catch {torch::group_norm 2} msg
    expr {[string match "*Invalid*" $msg] || [string match "*Wrong*" $msg]}
} {1}

test group_norm-6.3 {Error handling - invalid numGroups} {
    catch {torch::group_norm -numGroups 0 -numChannels 4} msg
    expr {[string match "*numGroups*" $msg] || [string match "*Invalid*" $msg]}
} {1}

test group_norm-6.4 {Error handling - invalid numChannels} {
    catch {torch::group_norm -numGroups 2 -numChannels 0} msg
    expr {[string match "*numChannels*" $msg] || [string match "*Invalid*" $msg]}
} {1}

test group_norm-6.5 {Error handling - negative numGroups} {
    catch {torch::group_norm -numGroups -1 -numChannels 4} msg
    expr {[string match "*numGroups*" $msg] || [string match "*Invalid*" $msg]}
} {1}

test group_norm-6.6 {Error handling - negative numChannels} {
    catch {torch::group_norm -numGroups 2 -numChannels -1} msg
    expr {[string match "*numChannels*" $msg] || [string match "*Invalid*" $msg]}
} {1}

test group_norm-6.7 {Error handling - unknown parameter} {
    catch {torch::group_norm -numGroups 2 -numChannels 4 -unknown_param value} msg
    expr {[string match "*Unknown*" $msg] || [string match "*Invalid*" $msg]}
} {1}

test group_norm-6.8 {Error handling - non-numeric parameters} {
    catch {torch::group_norm -numGroups "not_a_number" -numChannels 4} msg
    expr {[string match "*Invalid*" $msg] || [string match "*expected*" $msg]}
} {1}

test group_norm-6.9 {Error handling - non-numeric eps} {
    catch {torch::group_norm -numGroups 2 -numChannels 4 -eps "not_a_number"} msg
    expr {[string match "*Invalid*" $msg] || [string match "*expected*" $msg]}
} {1}

# Test different configurations
test group_norm-7.1 {Different configurations - typical usage} {
    set layer [torch::group_norm 8 16]
    set layer_valid [expr {[string match "groupnorm*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test group_norm-7.2 {Different configurations - different eps values} {
    set layer [torch::group_norm 2 4 1e-8]
    set layer_valid [expr {[string match "groupnorm*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

# Test edge cases
test group_norm-8.1 {Edge case - single group} {
    set layer [torch::group_norm 1 4]
    set layer_valid [expr {[string match "groupnorm*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test group_norm-8.2 {Edge case - groups equal channels} {
    set layer [torch::group_norm 4 4]
    set layer_valid [expr {[string match "groupnorm*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test group_norm-8.3 {Edge case - large number of groups} {
    set layer [torch::group_norm 16 32]
    set layer_valid [expr {[string match "groupnorm*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test group_norm-8.4 {Edge case - very small eps} {
    set layer [torch::group_norm 2 4 1e-10]
    set layer_valid [expr {[string match "groupnorm*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test group_norm-8.5 {Edge case - large eps} {
    set layer [torch::group_norm 2 4 1e-1]
    set layer_valid [expr {[string match "groupnorm*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

cleanupTests
