# torch::layer_norm

Create a Layer Normalization module for neural networks.

## Syntax

### Current Syntax (Backward Compatible)
```tcl
torch::layer_norm normalized_shape ?eps?
```

### New Syntax (Named Parameters)
```tcl
torch::layer_norm -normalizedShape normalized_shape ?-eps eps?
```

### camelCase Alias
```tcl
torch::layerNorm normalized_shape ?eps?
torch::layerNorm -normalizedShape normalized_shape ?-eps eps?
```

## Parameters

### Named Parameters
- **`-normalizedShape`** (int or list, required): Input shape from an expected input of size. Can be:
  - Single integer: Apply normalization over the last dimension
  - List of integers: Apply normalization over the last len(normalized_shape) dimensions
- **`-eps`** (double, optional): A value added to the denominator for numerical stability (default: 1e-5)

### Positional Parameters  
- **`normalized_shape`** (int or list, required): Input shape from an expected input of size
- **`eps`** (double, optional): A value added to the denominator for numerical stability (default: 1e-5)

## Description

The `torch::layer_norm` command creates a Layer Normalization module that applies normalization over the last certain number of dimensions. Layer normalization normalizes inputs across the features instead of the batch dimension, making it particularly useful for sequence models, transformers, and scenarios where batch size varies.

Key characteristics:
- **Feature-wise normalization**: Normalizes across features, not batch dimension
- **Training mode independent**: Works consistently in both training and evaluation
- **Learnable parameters**: Includes learnable scale (weight) and shift (bias) parameters
- **Numerical stability**: Uses epsilon to prevent division by zero

The command supports both the original positional syntax for backward compatibility and the new named parameter syntax for improved readability and flexibility.

## Return Value

Returns a layer normalization module handle that can be used with neural network operations and layer forwarding.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create layer norm for transformer-like models
set layer_norm [torch::layer_norm 512]  ;# d_model = 512

# Create layer norm for 2D features
set layer_norm_2d [torch::layer_norm {64 64}]  ;# Normalize over last 2 dimensions

# Create layer norm with custom epsilon
set layer_norm_eps [torch::layer_norm 768 1e-6]
```

### Named Parameter Syntax
```tcl
# Create layer norm with named parameters
set layer_norm [torch::layer_norm -normalizedShape 512]

# Create layer norm with custom epsilon
set layer_norm [torch::layer_norm -normalizedShape 768 -eps 1e-6]

# Alternative parameter order
set layer_norm [torch::layer_norm -eps 1e-6 -normalizedShape 768]
```

### camelCase Alias
```tcl
# Create layer norm using camelCase alias
set layer_norm [torch::layerNorm 512]
set layer_norm [torch::layerNorm -normalizedShape 512 -eps 1e-5]
```

### Transformer Architecture Usage
```tcl
# BERT-like architecture
set d_model 768
set layer_norm_1 [torch::layer_norm $d_model]
set layer_norm_2 [torch::layer_norm $d_model]

# GPT-like architecture  
set embed_dim 1024
set layer_norm [torch::layer_norm $embed_dim]

# Vision Transformer (ViT)
set patch_embed_dim 768
set layer_norm [torch::layer_norm $patch_embed_dim]
```

### Multi-Dimensional Normalization
```tcl
# Normalize over last dimension (typical for sequences)
set ln_1d [torch::layer_norm 512]  ;# For inputs: [batch, seq_len, 512]

# Normalize over last 2 dimensions  
set ln_2d [torch::layer_norm {32 32}]  ;# For inputs: [batch, height, 32, 32]

# Normalize over last 3 dimensions
set ln_3d [torch::layer_norm {16 16 16}]  ;# For inputs: [batch, depth, 16, 16, 16]
```

### Different Model Architectures
```tcl
# BERT models
set bert_base [torch::layer_norm 768]    ;# BERT-Base
set bert_large [torch::layer_norm 1024]  ;# BERT-Large

# GPT models
set gpt_small [torch::layer_norm 768]    ;# GPT-2 Small
set gpt_medium [torch::layer_norm 1024]  ;# GPT-2 Medium  
set gpt_large [torch::layer_norm 1280]   ;# GPT-2 Large
set gpt_xl [torch::layer_norm 1600]      ;# GPT-2 XL

# T5 models
set t5_small [torch::layer_norm 512]     ;# T5-Small
set t5_base [torch::layer_norm 768]      ;# T5-Base
set t5_large [torch::layer_norm 1024]    ;# T5-Large
```

### Custom Epsilon Values
```tcl
# Very stable normalization (larger epsilon)
set stable_ln [torch::layer_norm -normalizedShape 512 -eps 1e-4]

# High precision normalization (smaller epsilon)
set precise_ln [torch::layer_norm -normalizedShape 512 -eps 1e-8]

# Research/experimental settings
set experimental_ln [torch::layer_norm -normalizedShape 512 -eps 1e-10]
```

### Building Complete Models
```tcl
# Simple transformer encoder layer components
set d_model 512
set layer_norm_1 [torch::layer_norm $d_model]  ;# Post-attention norm
set layer_norm_2 [torch::layer_norm $d_model]  ;# Post-feed-forward norm

# Multi-head attention (simplified)
set attention [torch::multihead_attention $d_model 8]  ;# 8 attention heads
set feed_forward [torch::sequential [list \
    [torch::linear $d_model 2048] \
    [torch::linear 2048 $d_model] \
]]

# Transformer encoder layer (simplified usage pattern)
proc transformer_encoder_layer {input layer_norm_1 attention layer_norm_2 feed_forward} {
    # Self-attention with residual connection and layer norm
    set attn_output [torch::layer_forward $attention $input]
    set attn_residual [torch::tensor_add $input $attn_output]
    set norm1_output [torch::layer_forward $layer_norm_1 $attn_residual]
    
    # Feed-forward with residual connection and layer norm
    set ff_output [torch::layer_forward $feed_forward $norm1_output]
    set ff_residual [torch::tensor_add $norm1_output $ff_output]
    set final_output [torch::layer_forward $layer_norm_2 $ff_residual]
    
    return $final_output
}
```

### Research and Experimental Settings
```tcl
# Large language model dimensions
set llm_small [torch::layer_norm 2048]    ;# Small LLM
set llm_medium [torch::layer_norm 4096]   ;# Medium LLM
set llm_large [torch::layer_norm 8192]    ;# Large LLM
set llm_xl [torch::layer_norm 12288]      ;# Extra Large LLM

# Vision model dimensions
set vit_tiny [torch::layer_norm 192]      ;# ViT-Tiny
set vit_small [torch::layer_norm 384]     ;# ViT-Small
set vit_base [torch::layer_norm 768]      ;# ViT-Base
set vit_large [torch::layer_norm 1024]    ;# ViT-Large
set vit_huge [torch::layer_norm 1280]     ;# ViT-Huge
```

### Batch Processing and Efficiency
```tcl
# Create layer norm for different batch sizes (same parameters)
set layer_norm [torch::layer_norm 768]

# Example input shapes this layer norm can handle:
# - Single sequence: [1, seq_len, 768]
# - Batch processing: [32, seq_len, 768]  
# - Large batch: [128, seq_len, 768]
# - Variable sequence lengths work with appropriate masking
```

### Integration with Other Layers
```tcl
# Complete MLP with layer normalization
set d_model 512
set hidden_dim 2048

set pre_norm [torch::layer_norm $d_model]
set mlp [torch::sequential [list \
    [torch::linear $d_model $hidden_dim] \
    [torch::relu] \
    [torch::linear $hidden_dim $d_model] \
]]
set post_norm [torch::layer_norm $d_model]

# Usage pattern with residual connections
proc mlp_block {input pre_norm mlp post_norm} {
    set norm_input [torch::layer_forward $pre_norm $input]
    set mlp_output [torch::layer_forward $mlp $norm_input]
    set residual [torch::tensor_add $input $mlp_output]
    set final_output [torch::layer_forward $post_norm $residual]
    return $final_output
}
```

## Error Handling

The command performs comprehensive error checking:

```tcl
# Empty normalized shape
catch {torch::layer_norm {}} error
puts $error  ;# "Invalid normalized_shape value" or similar

# Invalid shape type
catch {torch::layer_norm invalid_shape} error
puts $error  ;# "Invalid normalized_shape value"

# Missing parameter value
catch {torch::layer_norm -normalizedShape} error
puts $error  ;# "Missing value for parameter"

# Unknown parameter
catch {torch::layer_norm -unknown_param value -normalizedShape 512} error
puts $error  ;# "Unknown parameter: -unknown_param"

# Invalid eps value (non-numeric)
catch {torch::layer_norm -normalizedShape 512 -eps invalid} error
puts $error  ;# "Invalid eps value"
```

## Normalized Shape Specifications

### Single Dimension (Most Common)
```tcl
# Normalize over the last dimension
set ln [torch::layer_norm 512]
# Input shape: [batch_size, seq_len, 512] 
# Normalization applied over the 512 features
```

### Multiple Dimensions
```tcl
# Normalize over last 2 dimensions
set ln [torch::layer_norm {64 64}]
# Input shape: [batch_size, channels, 64, 64]
# Normalization applied over the 64x64 spatial dimensions

# Normalize over last 3 dimensions  
set ln [torch::layer_norm {32 16 8}]
# Input shape: [batch_size, depth, 32, 16, 8]
# Normalization applied over the 32x16x8 volume
```

### Shape Requirements
- **Normalized shape** must match the last dimensions of input tensors
- **Input tensors** must have at least as many dimensions as normalized_shape
- **Batch dimension** is always preserved (never normalized)

## Epsilon Parameter Guidelines

### Default Value (1e-5)
```tcl
set ln [torch::layer_norm 512]  ;# Uses default eps=1e-5
```

### Common Values
```tcl
# Standard stable training
set ln [torch::layer_norm 512 1e-5]    ;# Default, good for most cases

# Higher stability (less precision)
set ln [torch::layer_norm 512 1e-4]    ;# More stable, less precise

# Higher precision (less stability)  
set ln [torch::layer_norm 512 1e-6]    ;# More precise, potentially less stable

# Research settings
set ln [torch::layer_norm 512 1e-8]    ;# Very precise, use with caution
```

### Choosing Epsilon
- **Larger epsilon**: More numerically stable, less precise normalization
- **Smaller epsilon**: More precise normalization, potential numerical issues
- **Default (1e-5)**: Good balance for most applications
- **Mixed precision**: May require larger epsilon (1e-4)

## Performance Considerations

### Memory Usage
```tcl
# Layer norm parameters scale with normalized_shape
set small_ln [torch::layer_norm 256]   ;# 512 parameters (256*2)
set large_ln [torch::layer_norm 1024]  ;# 2048 parameters (1024*2)

# Multi-dimensional shapes
set ln_2d [torch::layer_norm {64 64}]  ;# 8192 parameters (64*64*2)
```

### Computational Efficiency
```tcl
# Layer norm is generally efficient
# Computation scales linearly with normalized dimensions
# No dependency on batch size (unlike batch normalization)
```

## Input Shape Compatibility

### Sequence Models
```tcl
set layer_norm [torch::layer_norm 512]
# Compatible input shapes:
# - [batch_size, seq_len, 512]      # Standard sequence
# - [1, seq_len, 512]               # Single sequence
# - [batch_size, 1, 512]            # Single token
```

### Vision Models  
```tcl
set layer_norm [torch::layer_norm {16 16}]
# Compatible input shapes:
# - [batch_size, channels, 16, 16]  # Feature maps
# - [batch_size, 256]               # Flattened (16*16=256)
```

### Flexible Usage
```tcl
set layer_norm [torch::layer_norm 768]
# Works with various input shapes:
# - [32, 100, 768]    # Batch=32, seq_len=100
# - [1, 512, 768]     # Batch=1, seq_len=512  
# - [64, 50, 768]     # Batch=64, seq_len=50
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set layer_norm [torch::layer_norm 512]
set layer_norm [torch::layer_norm 512 1e-6]

# New named parameter syntax
set layer_norm [torch::layer_norm -normalizedShape 512]
set layer_norm [torch::layer_norm -normalizedShape 512 -eps 1e-6]
```

### From snake_case to camelCase

```tcl
# Old snake_case command
set layer_norm [torch::layer_norm 512]

# New camelCase command  
set layer_norm [torch::layerNorm 512]
```

## Integration with Training

### Training Loop Integration
```tcl
# Create model with layer normalization
set model_layers [list \
    [torch::linear 784 512] \
    [torch::layer_norm 512] \
    [torch::relu] \
    [torch::linear 512 256] \
    [torch::layer_norm 256] \
    [torch::relu] \
    [torch::linear 256 10] \
]

set model [torch::sequential $model_layers]

# Training loop (simplified)
for {set epoch 0} {$epoch < 10} {incr epoch} {
    # Forward pass (layer norm automatically handles training/eval modes)
    set predictions [torch::layer_forward $model $train_input]
    
    # Compute loss and backpropagate
    set loss [torch::mse_loss $predictions $train_targets]
    torch::tensor_backward $loss
    
    # Update parameters (including layer norm scale and bias)
    torch::optimizer_step $optimizer
}
```

### Parameter Access
```tcl
# Layer norm modules have learnable parameters
set layer_norm [torch::layer_norm 512]

# Parameters include:
# - weight (scale): initialized to ones
# - bias (shift): initialized to zeros
# These are automatically included in model parameters for optimization
```

## Best Practices

### Model Architecture Design
```tcl
# Good: Layer norm after linear transformations
set encoder_layer [torch::sequential [list \
    [torch::linear 512 512] \
    [torch::layer_norm 512] \
    [torch::relu] \
    [torch::linear 512 512] \
    [torch::layer_norm 512] \
]]

# Common pattern: Pre-norm vs Post-norm
# Pre-norm (more stable training)
proc pre_norm_layer {input layer_norm linear} {
    set norm_input [torch::layer_forward $layer_norm $input]
    set linear_output [torch::layer_forward $linear $norm_input]
    return [torch::tensor_add $input $linear_output]  ;# Residual connection
}

# Post-norm (original transformer design)
proc post_norm_layer {input linear layer_norm} {
    set linear_output [torch::layer_forward $linear $input]
    set residual [torch::tensor_add $input $linear_output]
    return [torch::layer_forward $layer_norm $residual]
}
```

### Debugging and Monitoring
```tcl
proc debug_layer_norm {layer_norm input} {
    puts "Layer norm module: $layer_norm"
    puts "Input shape: [torch::tensor_shape $input]"
    
    # Forward pass
    set output [torch::layer_forward $layer_norm $input]
    puts "Output shape: [torch::tensor_shape $output]"
    
    # Check normalization properties (mean ≈ 0, std ≈ 1)
    set mean [torch::tensor_mean $output -dim -1]
    set std [torch::tensor_std $output -dim -1]
    puts "Output mean (should be ~0): [torch::tensor_item $mean]"
    puts "Output std (should be ~1): [torch::tensor_item $std]"
    
    return $output
}

# Usage
set layer_norm [torch::layer_norm 512]
set input [torch::randn -shape {32 100 512}]
set output [debug_layer_norm $layer_norm $input]
```

### Memory Management
```tcl
# Layer norm modules persist until explicitly cleaned up
# They maintain learnable parameters (weight and bias tensors)
# Memory usage is typically small compared to linear layers

# For very large models, consider the total parameter count
proc count_layer_norm_params {normalized_shape} {
    if {[llength $normalized_shape] == 1} {
        return [expr {2 * $normalized_shape}]  ;# weight + bias
    } else {
        set total 1
        foreach dim $normalized_shape {
            set total [expr {$total * $dim}]
        }
        return [expr {2 * $total}]  ;# weight + bias
    }
}

# Example
set params [count_layer_norm_params 768]
puts "Layer norm parameters: $params"  ;# 1536 parameters
```

## Compatibility

- **Backward Compatible**: All existing positional syntax continues to work
- **Forward Compatible**: New named parameter syntax is preferred for new code  
- **Alias Support**: camelCase aliases provide modern API style
- **Error Handling**: Comprehensive validation with clear error messages
- **Cross-platform**: Works on all supported PyTorch platforms

## Version History

- **v1.0**: Original positional syntax implementation
- **v2.0**: Added dual syntax support with named parameters and camelCase aliases

## Common Use Cases

### 1. Transformer Models
```tcl
# Standard transformer setup
set d_model 512
set n_heads 8
set d_ff 2048

set layer_norm_1 [torch::layer_norm $d_model]  ;# Post-attention
set layer_norm_2 [torch::layer_norm $d_model]  ;# Post-feed-forward
```

### 2. BERT-style Models
```tcl
# BERT configuration
set hidden_size 768
set layer_norm [torch::layer_norm $hidden_size]
```

### 3. Vision Transformers
```tcl
# ViT configuration
set patch_embed_dim 768
set layer_norm [torch::layer_norm $patch_embed_dim]
```

### 4. Language Models
```tcl
# GPT-style language model
set embed_dim 1024
set layer_norm [torch::layer_norm $embed_dim]
```

### 5. Custom Architectures
```tcl
# Research/custom models
set custom_dim 1536
set layer_norm [torch::layer_norm $custom_dim 1e-6]  ;# Custom epsilon
```

## Related Commands

- [torch::batch_norm2d](batch_norm2d.md) - Batch normalization for 2D inputs
- [torch::group_norm](group_norm.md) - Group normalization
- [torch::instance_norm2d](instance_norm2d.md) - Instance normalization
- [torch::layer_forward](layer_forward.md) - Forward pass through layers
- [torch::linear](linear.md) - Linear/fully-connected layers

## See Also

- [Neural Network Guide](../guides/neural_networks.md)
- [Transformer Architecture Guide](../guides/transformers.md)  
- [Training Best Practices](../guides/training.md)
- [Model Optimization](../guides/optimization.md)
- [PyTorch Layer Normalization Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) 