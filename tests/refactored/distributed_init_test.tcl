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
# TORCH::DISTRIBUTED_INIT Tests - Dual Syntax Support
# ============================================================================

# Test 1: Basic positional syntax (backward compatibility)
test distributed_init-1.1 {Basic positional syntax - single GPU} {
    set result [torch::distributed_init 0 1 "127.0.0.1"]
    expr {[string match "*Distributed training initialized*rank=0*world_size=1*" $result]}
} {1}

test distributed_init-1.2 {Positional syntax with master port} {
    set result [torch::distributed_init 0 1 "127.0.0.1" 29501]
    expr {[string match "*Distributed training initialized*rank=0*world_size=1*" $result]}
} {1}

test distributed_init-1.3 {Positional syntax with master port and backend} {
    set result [torch::distributed_init 0 1 "127.0.0.1" 29502 "gloo"]
    expr {[string match "*Distributed training initialized*rank=0*world_size=1*backend=gloo*" $result]}
} {1}

test distributed_init-1.4 {Positional syntax - multi GPU (emulated)} {
    set result [torch::distributed_init 0 2 "127.0.0.1"]
    expr {[string match "*emulated multi-GPU*rank=0*world_size=2*" $result]}
} {1}

test distributed_init-1.5 {Positional syntax - different rank} {
    set result [torch::distributed_init 1 3 "192.168.1.100"]
    expr {[string match "*rank=1*world_size=3*" $result]}
} {1}

# Test 2: Named parameter syntax
test distributed_init-2.1 {Named parameter syntax - basic} {
    set result [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1"]
    expr {[string match "*rank=0*world_size=1*" $result]}
} {1}

test distributed_init-2.2 {Named parameter syntax with master port} {
    set result [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -masterPort 29503]
    expr {[string match "*rank=0*world_size=1*" $result]}
} {1}

test distributed_init-2.3 {Named parameter syntax with all parameters} {
    set result [torch::distributed_init -rank 1 -worldSize 4 -masterAddr "192.168.1.1" -masterPort 29504 -backend "gloo"]
    expr {[string match "*rank=1*world_size=4*backend=emulated_gloo*" $result]}
} {1}

test distributed_init-2.4 {Named parameter syntax - different parameter order} {
    set result [torch::distributed_init -worldSize 2 -rank 0 -masterAddr "localhost"]
    expr {[string match "*rank=0*world_size=2*" $result]}
} {1}

test distributed_init-2.5 {Named parameter syntax with backend only} {
    set result [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -backend "nccl"]
    expr {[string match "*backend=nccl*" $result]}
} {1}

test distributed_init-2.6 {Named parameter syntax with port only} {
    set result [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -masterPort 12345]
    expr {[string match "*rank=0*world_size=1*" $result]}
} {1}

# Test 3: camelCase alias tests
test distributed_init-3.1 {camelCase alias - basic} {
    set result [torch::distributedInit -rank 0 -worldSize 1 -masterAddr "127.0.0.1"]
    expr {[string match "*rank=0*world_size=1*" $result]}
} {1}

test distributed_init-3.2 {camelCase alias with all parameters} {
    set result [torch::distributedInit -rank 2 -worldSize 5 -masterAddr "10.0.0.1" -masterPort 29505 -backend "gloo"]
    expr {[string match "*rank=2*world_size=5*backend=emulated_gloo*" $result]}
} {1}

test distributed_init-3.3 {camelCase alias positional syntax} {
    set result [torch::distributedInit 0 1 "127.0.0.1" 29506]
    expr {[string match "*rank=0*world_size=1*" $result]}
} {1}

# Test 4: Parameter validation
test distributed_init-4.1 {Valid rank parameters} {
    set result1 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1"]
    set result2 [torch::distributed_init -rank 5 -worldSize 6 -masterAddr "127.0.0.1"]
    set result3 [torch::distributed_init -rank 10 -worldSize 11 -masterAddr "127.0.0.1"]
    expr {[string match "*rank=0*" $result1] && [string match "*rank=5*" $result2] && [string match "*rank=10*" $result3]}
} {1}

test distributed_init-4.2 {Valid world size parameters} {
    set result1 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1"]
    set result2 [torch::distributed_init -rank 0 -worldSize 8 -masterAddr "127.0.0.1"]
    set result3 [torch::distributed_init -rank 0 -worldSize 16 -masterAddr "127.0.0.1"]
    expr {[string match "*world_size=1*" $result1] && [string match "*world_size=8*" $result2] && [string match "*world_size=16*" $result3]}
} {1}

test distributed_init-4.3 {Valid master addresses} {
    set result1 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1"]
    set result2 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "localhost"]
    set result3 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "192.168.1.100"]
    set result4 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "node1.cluster.com"]
    expr {[string match "*Distributed training initialized*" $result1] && [string match "*Distributed training initialized*" $result2] && [string match "*Distributed training initialized*" $result3] && [string match "*Distributed training initialized*" $result4]}
} {1}

test distributed_init-4.4 {Valid master ports} {
    set result1 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -masterPort 29500]
    set result2 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -masterPort 8080]
    set result3 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -masterPort 65535]
    expr {[string match "*Distributed training initialized*" $result1] && [string match "*Distributed training initialized*" $result2] && [string match "*Distributed training initialized*" $result3]}
} {1}

test distributed_init-4.5 {Valid backends} {
    set result1 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -backend "nccl"]
    set result2 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -backend "gloo"]
    set result3 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -backend "mpi"]
    expr {[string match "*backend=nccl*" $result1] && [string match "*backend=gloo*" $result2] && [string match "*backend=mpi*" $result3]}
} {1}

# Test 5: Single vs Multi GPU behavior
test distributed_init-5.1 {Single GPU behavior} {
    set result [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1"]
    expr {[string match "*single GPU*" $result]}
} {1}

test distributed_init-5.2 {Multi GPU behavior (emulated)} {
    set result [torch::distributed_init -rank 0 -worldSize 2 -masterAddr "127.0.0.1"]
    expr {[string match "*emulated multi-GPU*" $result]}
} {1}

test distributed_init-5.3 {Multi GPU with different world sizes} {
    set result1 [torch::distributed_init -rank 0 -worldSize 2 -masterAddr "127.0.0.1"]
    set result2 [torch::distributed_init -rank 1 -worldSize 4 -masterAddr "127.0.0.1"]
    set result3 [torch::distributed_init -rank 3 -worldSize 8 -masterAddr "127.0.0.1"]
    expr {[string match "*world_size=2*" $result1] && [string match "*world_size=4*" $result2] && [string match "*world_size=8*" $result3]}
} {1}

# Test 6: Error handling tests
test distributed_init-6.1 {Missing required parameters in named syntax} {
    catch {torch::distributed_init -rank 0 -worldSize 1} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test distributed_init-6.2 {Missing rank parameter} {
    catch {torch::distributed_init -worldSize 1 -masterAddr "127.0.0.1"} result
    expr {[string match "*Required parameters missing*" $result] || [string match "*Missing value for parameter*" $result]}
} {1}

test distributed_init-6.3 {Missing world size parameter} {
    catch {torch::distributed_init -rank 0 -masterAddr "127.0.0.1"} result
    expr {[string match "*Required parameters missing*" $result] || [string match "*Missing value for parameter*" $result]}
} {1}

test distributed_init-6.4 {Missing master address parameter} {
    catch {torch::distributed_init -rank 0 -worldSize 1} result
    expr {[string match "*Required parameters missing*" $result] || [string match "*Missing value for parameter*" $result]}
} {1}

test distributed_init-6.5 {Invalid rank parameter type} {
    catch {torch::distributed_init -rank "not_a_number" -worldSize 1 -masterAddr "127.0.0.1"} result
    expr {[string match "*Invalid -rank parameter*" $result]}
} {1}

test distributed_init-6.6 {Invalid world size parameter type} {
    catch {torch::distributed_init -rank 0 -worldSize "not_a_number" -masterAddr "127.0.0.1"} result
    expr {[string match "*Invalid -worldSize parameter*" $result]}
} {1}

test distributed_init-6.7 {Invalid master port parameter type} {
    catch {torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -masterPort "not_a_number"} result
    expr {[string match "*Invalid -masterPort parameter*" $result]}
} {1}

test distributed_init-6.8 {Unknown parameter} {
    catch {torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -unknown_param value} result
    expr {[string match "*Unknown parameter: -unknown_param*" $result]}
} {1}

test distributed_init-6.9 {Missing value for parameter} {
    catch {torch::distributed_init -rank 0 -worldSize 1 -masterAddr} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

test distributed_init-6.10 {Wrong number of positional arguments - too few} {
    catch {torch::distributed_init 0 1} result
    expr {[string match "*Wrong number of arguments*" $result]}
} {1}

test distributed_init-6.11 {Wrong number of positional arguments - too many} {
    catch {torch::distributed_init 0 1 "127.0.0.1" 29500 "nccl" extra_arg} result
    expr {[string match "*Wrong number of arguments*" $result]}
} {1}

test distributed_init-6.12 {Negative rank} {
    catch {torch::distributed_init -rank -1 -worldSize 1 -masterAddr "127.0.0.1"} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test distributed_init-6.13 {Zero or negative world size} {
    catch {torch::distributed_init -rank 0 -worldSize 0 -masterAddr "127.0.0.1"} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test distributed_init-6.14 {Empty master address} {
    catch {torch::distributed_init -rank 0 -worldSize 1 -masterAddr ""} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

test distributed_init-6.15 {Zero or negative master port} {
    catch {torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -masterPort 0} result
    expr {[string match "*Required parameters missing*" $result]}
} {1}

# Test 7: Consistency between syntaxes
test distributed_init-7.1 {Positional vs named consistency - basic} {
    set result_pos [torch::distributed_init 0 1 "127.0.0.1"]
    set result_named [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1"]
    expr {[string match "*rank=0*world_size=1*" $result_pos] && [string match "*rank=0*world_size=1*" $result_named]}
} {1}

test distributed_init-7.2 {Positional vs named consistency - with port} {
    set result_pos [torch::distributed_init 1 2 "192.168.1.1" 29507]
    set result_named [torch::distributed_init -rank 1 -worldSize 2 -masterAddr "192.168.1.1" -masterPort 29507]
    expr {[string match "*rank=1*world_size=2*" $result_pos] && [string match "*rank=1*world_size=2*" $result_named]}
} {1}

test distributed_init-7.3 {Positional vs named consistency - full parameters} {
    set result_pos [torch::distributed_init 2 3 "10.0.0.1" 29508 "gloo"]
    set result_named [torch::distributed_init -rank 2 -worldSize 3 -masterAddr "10.0.0.1" -masterPort 29508 -backend "gloo"]
    expr {[string match "*rank=2*world_size=3*backend=*gloo*" $result_pos] && [string match "*rank=2*world_size=3*backend=*gloo*" $result_named]}
} {1}

test distributed_init-7.4 {snake_case vs camelCase consistency} {
    set result_snake [torch::distributed_init -rank 1 -worldSize 2 -masterAddr "127.0.0.1"]
    set result_camel [torch::distributedInit -rank 1 -worldSize 2 -masterAddr "127.0.0.1"]
    expr {[string match "*rank=1*world_size=2*" $result_snake] && [string match "*rank=1*world_size=2*" $result_camel]}
} {1}

# Test 8: Complex scenarios
test distributed_init-8.1 {Multiple initialization attempts} {
    set result1 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1"]
    set result2 [torch::distributed_init -rank 1 -worldSize 4 -masterAddr "192.168.1.10"]
    set result3 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "localhost"]
    expr {[string match "*rank=0*world_size=1*" $result1] && [string match "*rank=1*world_size=4*" $result2] && [string match "*rank=0*world_size=1*" $result3]}
} {1}

test distributed_init-8.2 {Mixed syntax usage} {
    set result1 [torch::distributed_init 0 2 "127.0.0.1"]
    set result2 [torch::distributedInit -rank 1 -worldSize 3 -masterAddr "127.0.0.1"]
    expr {[string match "*rank=0*world_size=2*" $result1] && [string match "*rank=1*world_size=3*" $result2]}
} {1}

test distributed_init-8.3 {Different parameter order variations} {
    set result1 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -masterPort 29509 -backend "nccl"]
    set result2 [torch::distributed_init -backend "gloo" -masterPort 29510 -masterAddr "127.0.0.1" -worldSize 1 -rank 0]
    set result3 [torch::distributed_init -masterAddr "127.0.0.1" -rank 0 -worldSize 1]
    expr {[string match "*rank=0*world_size=1*" $result1] && [string match "*rank=0*world_size=1*" $result2] && [string match "*rank=0*world_size=1*" $result3]}
} {1}

# Test 9: Edge cases and special values
test distributed_init-9.1 {Large rank and world size values} {
    set result1 [torch::distributed_init -rank 99 -worldSize 100 -masterAddr "127.0.0.1"]
    set result2 [torch::distributed_init -rank 255 -worldSize 256 -masterAddr "127.0.0.1"]
    expr {[string match "*rank=99*world_size=100*" $result1] && [string match "*rank=255*world_size=256*" $result2]}
} {1}

test distributed_init-9.2 {High port numbers} {
    set result1 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -masterPort 65534]
    set result2 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -masterPort 32768]
    expr {[string match "*Distributed training initialized*" $result1] && [string match "*Distributed training initialized*" $result2]}
} {1}

test distributed_init-9.3 {IPv6 addresses} {
    set result [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "::1"]
    expr {[string match "*Distributed training initialized*" $result]}
} {1}

test distributed_init-9.4 {Domain names} {
    set result1 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "localhost"]
    set result2 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "master.example.com"]
    expr {[string match "*Distributed training initialized*" $result1] && [string match "*Distributed training initialized*" $result2]}
} {1}

test distributed_init-9.5 {Special backend names} {
    set result1 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -backend "custom_backend"]
    set result2 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -backend "NCCL"]
    set result3 [torch::distributed_init -rank 0 -worldSize 1 -masterAddr "127.0.0.1" -backend ""]
    expr {[string match "*backend=custom_backend*" $result1] && [string match "*backend=NCCL*" $result2] && [string match "*Distributed training initialized*" $result3]}
} {1}

cleanupTests 