# torch::lstm

Create a Long Short-Term Memory (LSTM) recurrent neural network layer.

## 🔄 Dual Syntax Support

This command supports both legacy positional syntax and modern named parameter syntax.

### Named Parameters (Recommended)
```tcl
torch::lstm -inputSize $input_size -hiddenSize $hidden_size ?-numLayers $num_layers? ?-bias $bias? ?-batchFirst $batch_first? ?-dropout $dropout? ?-bidirectional $bidirectional?
torch::lstm -input_size $input_size -hidden_size $hidden_size ?-num_layers $num_layers? ?-bias $bias? ?-batch_first $batch_first? ?-dropout $dropout? ?-bidirectional $bidirectional?
```

### Positional Syntax (Legacy)
```tcl
torch::lstm $input_size $hidden_size ?$num_layers? ?$bias? ?$batch_first? ?$dropout? ?$bidirectional?
```

## 📖 Description

LSTM is a type of recurrent neural network (RNN) architecture that is well-suited for modeling sequences and handling the vanishing gradient problem. It maintains both short-term and long-term memory through a cell state and three gates: input, forget, and output gates.

## 🔧 Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-inputSize` / `-input_size` | integer | Yes | - | The number of expected features in the input |
| `-hiddenSize` / `-hidden_size` | integer | Yes | - | The number of features in the hidden state |
| `-numLayers` / `-num_layers` | integer | No | 1 | Number of recurrent layers |
| `-bias` | boolean | No | true | Whether to use bias parameters |
| `-batchFirst` / `-batch_first` | boolean | No | false | Whether input/output tensors are batch-first |
| `-dropout` | double | No | 0.0 | Dropout probability for outputs (except last layer) |
| `-bidirectional` | boolean | No | false | Whether to use bidirectional LSTM |

### Parameter Details

- **inputSize**: Size of input features (e.g., embedding dimension)
- **hiddenSize**: Size of hidden state (controls model capacity)
- **numLayers**: Depth of the LSTM (stacked layers)
- **bias**: Whether to include bias terms in linear transformations
- **batchFirst**: Input shape (batch_first=true: [batch, seq, feature], false: [seq, batch, feature])
- **dropout**: Dropout rate between layers (requires num_layers > 1)
- **bidirectional**: Process sequences in both directions

## 📊 Mathematical Background

LSTM processes sequences using the following gates and equations:

### Gate Equations
```
forget_gate = σ(W_f × [h_{t-1}, x_t] + b_f)
input_gate = σ(W_i × [h_{t-1}, x_t] + b_i)  
candidate = tanh(W_C × [h_{t-1}, x_t] + b_C)
output_gate = σ(W_o × [h_{t-1}, x_t] + b_o)
```

### State Updates
```
C_t = forget_gate ⊙ C_{t-1} + input_gate ⊙ candidate
h_t = output_gate ⊙ tanh(C_t)
```

Where:
- σ is the sigmoid function
- ⊙ is element-wise multiplication
- W and b are learned parameters
- C_t is the cell state, h_t is the hidden state

## 💡 Use Cases

1. **Sequence Modeling**: Natural language processing, time series prediction
2. **Language Models**: Text generation, machine translation
3. **Speech Recognition**: Converting audio to text
4. **Sentiment Analysis**: Classifying text sentiment
5. **Time Series**: Stock prices, weather prediction, sensor data

## 🎯 Examples

### Basic Usage with Named Parameters
```tcl
;# Simple LSTM for text processing
set lstm [torch::lstm -inputSize 300 -hiddenSize 128]

;# LSTM for time series with multiple layers
set lstm [torch::lstm -inputSize 10 -hiddenSize 64 -numLayers 2 -dropout 0.2]

;# Bidirectional LSTM for sequence classification
set lstm [torch::lstm -inputSize 100 -hiddenSize 256 -bidirectional true -batchFirst true]
```

### CamelCase Syntax (Already CamelCase)
```tcl
;# The command name is already in camelCase
set lstm [torch::lstm -inputSize 300 -hiddenSize 128 -numLayers 2]
```

### Snake_case Parameter Names
```tcl
;# Alternative parameter naming
set lstm [torch::lstm -input_size 300 -hidden_size 128 -num_layers 2 -batch_first true]
```

### Legacy Positional Syntax
```tcl
;# Backward compatible syntax
set lstm [torch::lstm 300 128]                        ;# Basic LSTM
set lstm [torch::lstm 300 128 2]                      ;# With num_layers
set lstm [torch::lstm 300 128 2 true]                 ;# With bias
set lstm [torch::lstm 300 128 2 true true]            ;# With batch_first
set lstm [torch::lstm 300 128 2 true false 0.2]       ;# With dropout
set lstm [torch::lstm 300 128 2 true false 0.1 true]  ;# All parameters
```

### Common NLP Patterns
```tcl
;# Word-level language model
set lstm [torch::lstm -inputSize 300 -hiddenSize 512 -numLayers 2 -dropout 0.3 -batchFirst true]

;# Character-level RNN
set lstm [torch::lstm -inputSize 128 -hiddenSize 256 -numLayers 3 -dropout 0.2]

;# Sequence-to-sequence encoder
set encoder [torch::lstm -inputSize 300 -hiddenSize 512 -bidirectional true -batchFirst true]

;# Sequence classifier
set classifier [torch::lstm -inputSize 768 -hiddenSize 256 -numLayers 1 -batchFirst true]
```

### Time Series Examples
```tcl
;# Stock price prediction
set lstm [torch::lstm -inputSize 5 -hiddenSize 64 -numLayers 2 -dropout 0.1]

;# Weather forecasting
set lstm [torch::lstm -inputSize 10 -hiddenSize 128 -numLayers 3 -bidirectional true]

;# Sensor data analysis
set lstm [torch::lstm -inputSize 20 -hiddenSize 32 -numLayers 1]
```

## 🔄 Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set lstm [torch::lstm 300 128 2 true false 0.2 true]
```

**New (Named Parameters):**
```tcl
set lstm [torch::lstm -inputSize 300 -hiddenSize 128 -numLayers 2 -bias true -batchFirst false -dropout 0.2 -bidirectional true]
```

**Benefits of Named Parameters:**
- **Clarity**: Each parameter's purpose is explicit
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Easy to modify specific parameters
- **Self-documenting**: Code is more readable and understandable

## ⚠️ Error Handling

The command provides detailed error messages for common issues:

```tcl
;# Missing required parameters
torch::lstm -inputSize 300
;# Error: input_size, hidden_size, and num_layers must be > 0

;# Invalid parameter values
torch::lstm -inputSize 0 -hiddenSize 128
;# Error: input_size, hidden_size, and num_layers must be > 0

torch::lstm -inputSize 300 -hiddenSize 128 -dropout -0.1
;# Error: dropout must be >= 0.0

;# Invalid parameter types
torch::lstm -inputSize invalid -hiddenSize 128
;# Error: Invalid input_size value

torch::lstm -inputSize 300 -hiddenSize 128 -bias invalid
;# Error: Invalid bias value

;# Unknown parameters
torch::lstm -inputSize 300 -hiddenSize 128 -unknown param
;# Error: Unknown parameter: -unknown
```

## 🎯 Best Practices

### Architecture Design
1. **Hidden Size**: Start with 128-512 for most tasks
2. **Layers**: 1-3 layers often sufficient; more may overfit
3. **Bidirectional**: Use for classification, avoid for generation
4. **Dropout**: 0.2-0.5 between layers to prevent overfitting

### Input/Output Configuration
1. **Batch First**: Recommended for most modern workflows
2. **Input Size**: Match your embedding or feature dimension
3. **Sequence Length**: Consider memory constraints for very long sequences

### Training Tips
1. **Gradient Clipping**: Often necessary to prevent exploding gradients
2. **Learning Rate**: Start with 0.001-0.01, may need lower rates
3. **Initialization**: Default PyTorch initialization usually works well

## 📈 Performance Considerations

- **Memory Usage**: O(layers × hidden_size²) for parameters
- **Computation**: Sequential nature limits parallelization within sequences
- **Bidirectional**: Doubles computation but often improves accuracy
- **Batch Size**: Larger batches improve GPU utilization

### Memory Optimization
```tcl
;# Memory-efficient for long sequences
set lstm [torch::lstm -inputSize 300 -hiddenSize 128 -numLayers 1]

;# Higher capacity for complex tasks
set lstm [torch::lstm -inputSize 300 -hiddenSize 512 -numLayers 2 -dropout 0.3]
```

## 🔗 Related Commands

- `torch::gru` - Gated Recurrent Unit (simpler alternative)
- `torch::rnn_tanh` - Vanilla RNN with tanh activation
- `torch::rnn_relu` - Vanilla RNN with ReLU activation
- `torch::linear` - For output projection layers
- `torch::embedding` - For input embeddings

## 📋 Return Value

Returns an LSTM layer handle (string) that can be used with:
- `torch::layer_forward` - Forward pass through the layer
- `torch::save_model` - Save the trained model
- `torch::load_model` - Load a saved model

### Handle Format
```tcl
set lstm [torch::lstm -inputSize 300 -hiddenSize 128]
puts $lstm  ;# Outputs: lstm0, lstm1, lstm2, etc.
```

## 🧪 Testing

The command includes comprehensive test coverage:
- ✅ Dual syntax parsing (56 test cases)
- ✅ Parameter validation and error handling
- ✅ Both positional and named parameter syntax
- ✅ CamelCase and snake_case parameter names
- ✅ Boolean parameter variations
- ✅ Edge cases and boundary conditions

## 📚 Technical Notes

### Implementation Details
- **Backend**: Uses PyTorch's native LSTM implementation
- **CUDA Support**: Automatically uses GPU when available
- **Precision**: Supports both float32 and float64
- **Thread Safety**: Layer creation is thread-safe

### Compatibility
- **Backward Compatibility**: 100% compatible with existing positional syntax
- **Parameter Flexibility**: Supports both camelCase and snake_case naming
- **PyTorch Version**: Compatible with PyTorch 1.x and 2.x

### Common Gotchas
1. **Dropout Warning**: PyTorch warns when dropout > 0 and num_layers = 1
2. **Batch Dimensions**: Pay attention to batch_first setting
3. **Hidden State**: Remember to handle initial hidden states in training loops
4. **Gradient Flow**: LSTM helps with vanishing gradients but exploding gradients may still occur

## 🔍 Example Workflows

### Text Classification
```tcl
;# 1. Create LSTM
set lstm [torch::lstm -inputSize 300 -hiddenSize 128 -batchFirst true]

;# 2. Create classification head
set classifier [torch::linear 128 2]  ;# Binary classification

;# 3. Forward pass (conceptual)
;# output = torch::layer_forward $lstm $embedded_text
;# logits = torch::layer_forward $classifier $output
```

### Sequence-to-Sequence
```tcl
;# Encoder
set encoder [torch::lstm -inputSize 300 -hiddenSize 512 -bidirectional true -batchFirst true]

;# Decoder  
set decoder [torch::lstm -inputSize 300 -hiddenSize 512 -batchFirst true]
```

### Time Series Prediction
```tcl
;# Multi-variate time series
set lstm [torch::lstm -inputSize 10 -hiddenSize 64 -numLayers 2 -dropout 0.1 -batchFirst true]

;# Output projection
set output_layer [torch::linear 64 1]  ;# Predict single value
``` 