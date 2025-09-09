#!/usr/bin/env tclsh

package require tcltest
namespace import tcltest::*

# Load the shared library
if {[catch {load ../../build/libtorchtcl.so} result]} {
    puts "Failed to load libtorchtcl.so: $result"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip start}

# Test suite for torch::distributed_recv command
test distributed_recv-1.1 {Basic positional syntax} {
    set tensor [torch::distributed_recv {2 3} 0]
    expr {$tensor ne ""}
} {1}

test distributed_recv-1.2 {Positional syntax with tag} {
    set tensor [torch::distributed_recv {3 4} 1 42]
    expr {$tensor ne ""}
} {1}

test distributed_recv-1.3 {Positional syntax with different src} {
    set tensor [torch::distributed_recv {5 2} 2]
    expr {$tensor ne ""}
} {1}

# Test named parameter syntax
test distributed_recv-2.1 {Named parameter syntax basic} {
    set tensor [torch::distributed_recv -shape {2 3} -src 0]
    expr {$tensor ne ""}
} {1}

test distributed_recv-2.2 {Named parameter syntax with tag} {
    set tensor [torch::distributed_recv -shape {3 4} -src 1 -tag 100]
    expr {$tensor ne ""}
} {1}

test distributed_recv-2.3 {Named parameter syntax different order} {
    set tensor [torch::distributed_recv -src 2 -shape {4 5} -tag 25]
    expr {$tensor ne ""}
} {1}

# Test camelCase alias
test distributed_recv-3.1 {camelCase alias basic} {
    set tensor [torch::distributedRecv {2 3} 0]
    expr {$tensor ne ""}
} {1}

test distributed_recv-3.2 {camelCase alias with named parameters} {
    set tensor [torch::distributedRecv -shape {3 4} -src 1 -tag 50]
    expr {$tensor ne ""}
} {1}

test distributed_recv-3.3 {camelCase alias with tag} {
    set tensor [torch::distributedRecv {4 2} 2 15]
    expr {$tensor ne ""}
} {1}

# Test command existence
test distributed_recv-4.1 {Verify torch::distributed_recv command exists} {
    info commands torch::distributed_recv
} {::torch::distributed_recv}

test distributed_recv-4.2 {Verify torch::distributedRecv camelCase alias exists} {
    info commands torch::distributedRecv
} {::torch::distributedRecv}

# Test different tensor shapes
test distributed_recv-5.1 {1D tensor} {
    set tensor [torch::distributed_recv {10} 0]
    set shape [torch::tensor_shape $tensor]
    set shape
} {10}

test distributed_recv-5.2 {2D tensor} {
    set tensor [torch::distributed_recv {3 4} 1]
    set shape [torch::tensor_shape $tensor]
    set shape
} {3 4}

test distributed_recv-5.3 {3D tensor} {
    set tensor [torch::distributed_recv {2 3 4} 0]
    set shape [torch::tensor_shape $tensor]
    set shape
} {2 3 4}

test distributed_recv-5.4 {4D tensor} {
    set tensor [torch::distributed_recv {2 3 4 5} 1]
    set shape [torch::tensor_shape $tensor]
    set shape
} {2 3 4 5}

# Test tensor content (zeros tensor)
test distributed_recv-6.1 {Tensor is valid non-empty handle} {
    set tensor [torch::distributed_recv {2 2} 0]
    expr {[string length $tensor] > 0}
} {1}

test distributed_recv-6.2 {Different shape creates valid tensor} {
    set tensor [torch::distributed_recv {1 3} 0]
    expr {[string length $tensor] > 0}
} {1}

# Error handling tests
test distributed_recv-7.1 {Missing arguments} {
    catch {torch::distributed_recv} error
    string match "*Required parameters missing*" $error
} {1}

test distributed_recv-7.2 {Missing src in positional syntax} {
    catch {torch::distributed_recv {2 3}} error
    string match "*Wrong number of arguments*" $error
} {1}

test distributed_recv-7.3 {Invalid shape format} {
    catch {torch::distributed_recv "invalid_shape" 0} error
    string match "*Invalid*" $error
} {1}

test distributed_recv-7.4 {Invalid src parameter} {
    catch {torch::distributed_recv {2 3} "invalid_src"} error
    string match "*Invalid src parameter*" $error
} {1}

test distributed_recv-7.5 {Invalid tag parameter} {
    catch {torch::distributed_recv {2 3} 0 "invalid_tag"} error
    string match "*Invalid tag parameter*" $error
} {1}

test distributed_recv-7.6 {Unknown named parameter} {
    catch {torch::distributed_recv -shape {2 3} -src 0 -unknown_param 1} error
    string match "*Unknown parameter*" $error
} {1}

test distributed_recv-7.7 {Missing value for named parameter} {
    catch {torch::distributed_recv -shape {2 3} -src} error
    string match "*Missing value for parameter*" $error
} {1}

test distributed_recv-7.8 {Missing required named parameters} {
    catch {torch::distributed_recv -shape {2 3}} error
    string match "*Required parameters missing*" $error
} {1}

test distributed_recv-7.9 {Invalid named src parameter} {
    catch {torch::distributed_recv -shape {2 3} -src "invalid"} error
    string match "*Invalid -src parameter*" $error
} {1}

test distributed_recv-7.10 {Invalid named tag parameter} {
    catch {torch::distributed_recv -shape {2 3} -src 0 -tag "invalid"} error
    string match "*Invalid -tag parameter*" $error
} {1}

test distributed_recv-7.11 {Empty shape} {
    catch {torch::distributed_recv {} 0} error
    string match "*Required parameters missing*" $error
} {1}

test distributed_recv-7.12 {Negative src parameter} {
    catch {torch::distributed_recv {2 3} -1} error
    string match "*Required parameters missing*" $error
} {1}

# Test parameter combinations
test distributed_recv-8.1 {Zero src} {
    set tensor [torch::distributed_recv {2 3} 0]
    expr {$tensor ne ""}
} {1}

test distributed_recv-8.2 {Large src value} {
    set tensor [torch::distributed_recv {2 3} 1000]
    expr {$tensor ne ""}
} {1}

test distributed_recv-8.3 {Large tag value} {
    set tensor [torch::distributed_recv {2 3} 0 999999]
    expr {$tensor ne ""}
} {1}

test distributed_recv-8.4 {Negative tag value} {
    set tensor [torch::distributed_recv {2 3} 0 -5]
    expr {$tensor ne ""}
} {1}

test distributed_recv-8.5 {Zero tag value} {
    set tensor [torch::distributed_recv {2 3} 0 0]
    expr {$tensor ne ""}
} {1}

# Test syntax equivalence
test distributed_recv-9.1 {Positional and named syntax equivalence} {
    set tensor1 [torch::distributed_recv {2 3} 1 5]
    set tensor2 [torch::distributed_recv -shape {2 3} -src 1 -tag 5]
    set shape1 [torch::tensor_shape $tensor1]
    set shape2 [torch::tensor_shape $tensor2]
    expr {$shape1 eq $shape2}
} {1}

test distributed_recv-9.2 {Named syntax parameter order independence} {
    set tensor1 [torch::distributed_recv -shape {2 3} -src 2 -tag 10]
    set tensor2 [torch::distributed_recv -src 2 -tag 10 -shape {2 3}]
    set shape1 [torch::tensor_shape $tensor1]
    set shape2 [torch::tensor_shape $tensor2]
    expr {$shape1 eq $shape2}
} {1}

test distributed_recv-9.3 {camelCase and snake_case equivalence} {
    set tensor1 [torch::distributed_recv {2 3} 1 5]
    set tensor2 [torch::distributedRecv {2 3} 1 5]
    set shape1 [torch::tensor_shape $tensor1]
    set shape2 [torch::tensor_shape $tensor2]
    expr {$shape1 eq $shape2}
} {1}

# Test tensor properties
test distributed_recv-10.1 {Tensor is created successfully} {
    set tensor [torch::distributed_recv {2 3} 0]
    expr {[string length $tensor] > 0}
} {1}

test distributed_recv-10.2 {Tensor handle is valid} {
    set tensor [torch::distributed_recv {2 3} 0]
    expr {[string length $tensor] > 0}
} {1}

test distributed_recv-10.3 {Tensor is properly initialized} {
    set tensor [torch::distributed_recv {3 3} 0]
    set numel [torch::tensor_numel $tensor]
    set numel
} {9}

# Test edge cases
test distributed_recv-11.1 {Single element tensor} {
    set tensor [torch::distributed_recv {1} 0]
    set shape [torch::tensor_shape $tensor]
    set shape
} {1}

test distributed_recv-11.2 {Large tensor} {
    set tensor [torch::distributed_recv {100 100} 0]
    set numel [torch::tensor_numel $tensor]
    set numel
} {10000}

test distributed_recv-11.3 {5D tensor} {
    set tensor [torch::distributed_recv {2 2 2 2 2} 0]
    set numel [torch::tensor_numel $tensor]
    set numel
} {32}

# Test multiple receives
test distributed_recv-12.1 {Multiple receives from same source} {
    set tensor1 [torch::distributed_recv {2 2} 0]
    set tensor2 [torch::distributed_recv {3 3} 0]
    expr {$tensor1 ne $tensor2}
} {1}

test distributed_recv-12.2 {Multiple receives with different tags} {
    set tensor1 [torch::distributed_recv {2 2} 0 10]
    set tensor2 [torch::distributed_recv {2 2} 0 20]
    expr {$tensor1 ne $tensor2}
} {1}

test distributed_recv-12.3 {Multiple receives from different sources} {
    set tensor1 [torch::distributed_recv {2 2} 0]
    set tensor2 [torch::distributed_recv {2 2} 1]
    expr {$tensor1 ne $tensor2}
} {1}

# Clean up
cleanupTests 