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

proc createParams {} {
    set w [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0} -shape {3 3} -dtype float32 -device cpu]
    set b [torch::tensor_create -data {10.0 11.0 12.0} -shape {3} -dtype float32 -device cpu]
    return [list $w $b]
}

# Positional syntax

test optimizer_rmsprop-1.1 {Positional basic} {
    set p [createParams]
    set o [torch::optimizer_rmsprop $p 0.01]
    expr {[string match "optimizer*" $o]}
} {1}

test optimizer_rmsprop-1.2 {Positional with alpha & eps} {
    set p [createParams]
    set o [torch::optimizer_rmsprop $p 0.01 0.95 1e-07]
    expr {[string match "optimizer*" $o]}
} {1}

# Named syntax

test optimizer_rmsprop-2.1 {Named basic} {
    set p [createParams]
    set o [torch::optimizer_rmsprop -parameters $p -lr 0.02]
    expr {[string match "optimizer*" $o]}
} {1}

test optimizer_rmsprop-2.2 {Named alpha/eps} {
    set p [createParams]
    set o [torch::optimizer_rmsprop -parameters $p -lr 0.02 -alpha 0.97 -eps 1e-08]
    expr {[string match "optimizer*" $o]}
} {1}

# CamelCase alias

test optimizer_rmsprop-3.1 {CamelCase positional} {
    set p [createParams]
    set o [torch::optimizerRmsprop $p 0.01]
    expr {[string match "optimizer*" $o]}
} {1}

test optimizer_rmsprop-3.2 {CamelCase named} {
    set p [createParams]
    set o [torch::optimizerRmsprop -parameters $p -lr 0.03]
    expr {[string match "optimizer*" $o]}
} {1}

# Error handling

test optimizer_rmsprop-4.1 {Missing lr positional} {
    set p [createParams]
    catch {torch::optimizer_rmsprop $p} res
    expr {[string match "*Usage: torch::optimizer_rmsprop*" $res]}
} {1}

test optimizer_rmsprop-4.2 {Invalid lr named} {
    set p [createParams]
    catch {torch::optimizer_rmsprop -parameters $p -lr -0.1} res
    expr {[string match "*Invalid learning rate*" $res] || [string match "*Required parameters missing*" $res]}
} {1}

cleanupTests 