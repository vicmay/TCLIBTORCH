#!/usr/bin/env tclsh

# Load the LibTorch TCL extension
load ./libtorchtcl.so

puts "=== Testing 18 New Commands (6 Optimizers + 12 Schedulers) ==="

# Create a simple model for testing
set linear [torch::linear 10 5]
puts "✓ Created linear layer: $linear"

# Test all 6 new optimizers
puts "\n=== Testing 6 New Optimizers ==="

# 1. Sparse Adam
set sparse_adam [torch::optimizer_sparse_adam $linear 0.001]
puts "✓ torch::optimizer_sparse_adam: $sparse_adam"

# 2. NAdam
set nadam [torch::optimizer_nadam $linear 0.002]
puts "✓ torch::optimizer_nadam: $nadam"

# 3. RAdam
set radam [torch::optimizer_radam $linear 0.001]
puts "✓ torch::optimizer_radam: $radam"

# 4. Adafactor
set adafactor [torch::optimizer_adafactor $linear 0.8]
puts "✓ torch::optimizer_adafactor: $adafactor"

# 5. LAMB
set lamb [torch::optimizer_lamb $linear 0.001]
puts "✓ torch::optimizer_lamb: $lamb"

# 6. NovoGrad
set novograd [torch::optimizer_novograd $linear 0.01]
puts "✓ torch::optimizer_novograd: $novograd"

# Test all 12 new learning rate schedulers
puts "\n=== Testing 12 New Learning Rate Schedulers ==="

# 1. Multiplicative LR
set mult_lr [torch::lr_scheduler_multiplicative $sparse_adam 0.95]
puts "✓ torch::lr_scheduler_multiplicative: $mult_lr"

# 2. Polynomial LR
set poly_lr [torch::lr_scheduler_polynomial $nadam 100 2.0]
puts "✓ torch::lr_scheduler_polynomial: $poly_lr"

# 3. Cosine Annealing Warm Restarts
set cosine_warm [torch::lr_scheduler_cosine_annealing_warm_restarts $radam 10]
puts "✓ torch::lr_scheduler_cosine_annealing_warm_restarts: $cosine_warm"

# 4. Linear with Warmup
set linear_warmup [torch::lr_scheduler_linear_with_warmup $adafactor 100 10]
puts "✓ torch::lr_scheduler_linear_with_warmup: $linear_warmup"

# 5. Constant with Warmup
set const_warmup [torch::lr_scheduler_constant_with_warmup $lamb 10]
puts "✓ torch::lr_scheduler_constant_with_warmup: $const_warmup"

# 6. MultiStep LR
set multistep [torch::lr_scheduler_multi_step $novograd {30 60 90}]
puts "✓ torch::lr_scheduler_multi_step: $multistep"

# 7. Cosine Annealing LR
set cosine_ann [torch::lr_scheduler_cosine_annealing $sparse_adam 100]
puts "✓ torch::lr_scheduler_cosine_annealing: $cosine_ann"

# 8. Plateau LR
set plateau [torch::lr_scheduler_plateau $nadam]
puts "✓ torch::lr_scheduler_plateau: $plateau"

# 9. Inverse Square Root LR
set inv_sqrt [torch::lr_scheduler_inverse_sqrt $radam 1000]
puts "✓ torch::lr_scheduler_inverse_sqrt: $inv_sqrt"

# 10. Noam LR
set noam [torch::lr_scheduler_noam $adafactor 512]
puts "✓ torch::lr_scheduler_noam: $noam"

# 11. One Cycle Advanced LR
set one_cycle_adv [torch::lr_scheduler_onecycle_advanced $lamb 0.1 100]
puts "✓ torch::lr_scheduler_onecycle_advanced: $one_cycle_adv"

# Create a simple SGD optimizer for the last scheduler test
set params [torch::layer_parameters $linear]
set sgd [torch::optimizer_sgd $params 0.01]

# Create a basic step scheduler first, then test step_advanced
set basic_step [torch::lr_scheduler_step $sgd 30 0.1]

# 12. Step Advanced LR (operates on existing scheduler)
set step_adv [torch::lr_scheduler_step_advanced $basic_step 0.5]
puts "✓ torch::lr_scheduler_step_advanced: $step_adv"

# Test some basic functionality
puts "\n=== Testing Basic Functionality ==="

# Test optimizer step
torch::optimizer_step $sparse_adam
puts "✓ Optimizer step executed successfully"

# Test scheduler step
torch::lr_scheduler_step_update $mult_lr
puts "✓ Scheduler step executed successfully"

# Test getting learning rate from optimizer
set current_lr [torch::get_lr $sparse_adam]
puts "✓ Current learning rate: $current_lr"

puts "\n=== All 18 Commands Successfully Tested! ==="
puts "✓ 6 New Optimizers: sparse_adam, nadam, radam, adafactor, lamb, novograd"
puts "✓ 12 New Schedulers: multiplicative, polynomial, cosine_warm_restarts, linear_warmup,"
puts "                     constant_warmup, multistep, cosine_annealing, plateau,"
puts "                     inverse_sqrt, noam, one_cycle_advanced, step_advanced"

puts "\n=== BATCH IMPLEMENTATION COMPLETE ==="
puts "Total commands added: 18"
puts "New command count: 294 + 18 = 312"
puts "Completion rate: ~62%" 