# EasiScriptX Syntax Reference

## Overview

EasiScriptX (ESX) is a declarative domain-specific language for AI/ML workflows. This document provides a complete syntax reference using Extended Backus-Naur Form (EBNF).

## Grammar Definition

### Program Structure

```
program = { statement } ;

statement = let_statement
          | train_statement
          | model_statement
          | distribute_statement
          | autonomic_statement
          | memory_optimization_statement
          | llm_optimization_statement
          | pattern_recognition_statement
          | energy_aware_statement
          | framework_statement
          | experiment_statement
          | custom_function_statement
          ;
```

### Variable Declaration

```
let_statement = "let" identifier "=" expression ";" ;

identifier = letter { letter | digit | "_" } ;

letter = "a" | "b" | ... | "z" | "A" | "B" | ... | "Z" ;

digit = "0" | "1" | ... | "9" ;
```

### Expressions

```
expression = tensor_literal
           | identifier
           | matrix_multiplication
           | convolution_expression
           | attention_expression
           | lora_expression
           | mixed_precision_expression
           | fused_operation_expression
           | quantization_expression
           | pruning_expression
           | rnn_expression
           | transformer_expression
           ;

tensor_literal = "tensor" "(" "[" { number { "," number } } "]" { "," "[" { number { "," number } } "]" } ")" ;

number = [ "-" ] digit { digit } [ "." digit { digit } ] ;
```

### Tensor Operations

```
matrix_multiplication = expression "@" expression ;

convolution_expression = "conv2d" "(" expression "," expression "," "stride:" integer "," "padding:" integer ")" ;

attention_expression = "attention" "(" expression "," expression "," expression "," "heads:" integer "," "dim:" integer [ "," "flash=true" ] ")" ;

lora_expression = "lora" "(" expression "," "rank:" integer ")" ;

mixed_precision_expression = "mixed_precision" "(" expression "," ( "bf16" | "fp16" ) ")" ;

fused_operation_expression = "fused_matmul_relu" "(" expression "," expression ")" ;

quantization_expression = "quantize" "(" expression "," "bits:" ( "8" | "4" ) "," "method:" ( "ptq" | "qat" ) ")" ;

pruning_expression = "prune" "(" expression "," "ratio:" number ")" ;

rnn_expression = "rnn" "(" expression "," "hidden_size:" integer "," "layers:" integer ")" ;

transformer_expression = "transformer_block" "(" expression "," "heads:" integer "," "dim:" integer ")" ;
```

### Training Statements

```
train_statement = "train" "(" expression "," expression "," "loss:" loss_function "," "opt:" optimizer "," "epochs:" integer "," "device:" device ")" ;

loss_function = "ce" | "mse" | "huber" | "hinge" ;

optimizer = "adam" "(" "lr=" number ")" 
          | "sgd" "(" "lr=" number ")" 
          | "lomo" "(" "lr=" number ")" 
          ;

device = "cpu" | "gpu" ;
```

### Model Definition

```
model_statement = "model" identifier "{" { statement } "}" ;
```

### Distributed Training

```
distribute_statement = "distribute" "(" "gpus:" integer ")" "{" { statement } "}" ;

pipeline_parallel_statement = "pipeline_parallel" "(" expression "," "stages:" integer ")" ;
```

### Autonomic Optimization

```
autonomic_statement = "with" "autonomic" "{" { statement } "}" ;
```

### Memory Optimization

```
memory_optimization_statement = memory_broker_statement
                               | quantization_statement
                               | checkpoint_statement
                               ;

memory_broker_statement = "memory_broker" "(" expression "," "max_mem:" number "," "strategy:" ( "zeRO" | "offload" ) ")" ;

quantization_statement = "quantize" "(" expression "," "bits:" ( "8" | "4" ) "," "method:" ( "ptq" | "qat" ) ")" ;

checkpoint_statement = "checkpoint" "(" expression "," "segments:" integer ")" ;
```

### LLM Optimization

```
llm_optimization_statement = speculative_decode_statement
                            | instruction_tune_statement
                            | domain_adapt_statement
                            ;

speculative_decode_statement = "speculative_decode" "(" expression "," expression "," "max_tokens:" integer ")" ;

instruction_tune_statement = "instruction_tune" "(" expression "," expression "," string_literal ")" ;

domain_adapt_statement = "adapt_domain" "(" expression "," domain_type ")" ;

domain_type = "scientific" | "medical" | "legal" | "financial" | "general" ;
```

### Pattern Recognition

```
pattern_recognition_statement = "pattern_recognize" "(" expression "," "rules:" pattern_rules ")" ;

pattern_rules = "geometric" | "arithmetic" ;
```

### Energy-Aware Computing

```
energy_aware_statement = "energy_aware" "(" expression "," "max_power:" number ")" ;

heterogeneous_schedule_statement = "schedule_heterogeneous" "(" "cpu:" number "," "gpu:" number ")" ;
```

### Framework Interoperability

```
framework_statement = "switch_framework" "(" framework_type "," string_literal ")" ;

framework_type = "pytorch" | "tensorflow" | "jax" ;
```

### Experiment Tracking

```
experiment_statement = "track_experiment" "(" "mlflow" "," string_literal ")" ;
```

### Custom Functions

```
custom_function_statement = "fn" identifier "(" { identifier { "," identifier } } ")" "{" "return" expression "}" ;
```

### String Literals

```
string_literal = '"' { character } '"' ;

character = letter | digit | " " | "!" | "#" | "$" | "%" | "&" | "'" | "(" | ")" | "*" | "+" | "," | "-" | "." | "/" | ":" | ";" | "<" | "=" | ">" | "?" | "@" | "[" | "\" | "]" | "^" | "_" | "`" | "{" | "|" | "}" | "~" ;
```

### Integer Literals

```
integer = digit { digit } ;
```

## Lexical Rules

### Comments

```
comment = "//" { character } newline
        | "/*" { character | newline } "*/"
        ;

newline = "\n" | "\r\n" | "\r" ;
```

### Whitespace

```
whitespace = " " | "\t" | newline ;
```

### Keywords

The following are reserved keywords in EasiScriptX:

- `let` - Variable declaration
- `model` - Model definition
- `train` - Training statement
- `distribute` - Distributed training
- `with` - Autonomic optimization
- `autonomic` - Autonomic optimization
- `fn` - Function definition
- `return` - Function return
- `tensor` - Tensor literal
- `conv2d` - Convolution operation
- `attention` - Attention mechanism
- `lora` - LoRA adaptation
- `mixed_precision` - Mixed precision
- `fused_matmul_relu` - Fused operation
- `quantize` - Quantization
- `prune` - Pruning
- `rnn` - RNN operation
- `transformer_block` - Transformer block
- `memory_broker` - Memory optimization
- `checkpoint` - Gradient checkpointing
- `speculative_decode` - Speculative decoding
- `instruction_tune` - Instruction tuning
- `adapt_domain` - Domain adaptation
- `pattern_recognize` - Pattern recognition
- `energy_aware` - Energy-aware computing
- `schedule_heterogeneous` - Heterogeneous scheduling
- `switch_framework` - Framework switching
- `track_experiment` - Experiment tracking
- `pipeline_parallel` - Pipeline parallelism

### Operators

- `@` - Matrix multiplication
- `=` - Assignment
- `+` - Addition
- `-` - Subtraction
- `*` - Multiplication
- `/` - Division
- `^` - Exponentiation
- `(` - Left parenthesis
- `)` - Right parenthesis
- `[` - Left bracket
- `]` - Right bracket
- `{` - Left brace
- `}` - Right brace
- `,` - Comma
- `:` - Colon
- `;` - Semicolon

## Examples

### Basic Tensor Operations

```esx
let x = tensor([[1,2],[3,4]])
let y = x @ x
let z = conv2d(x, x, stride:1, padding:0)
```

### Training Statement

```esx
train(model, dataset, loss: ce, opt: adam(lr=0.001), epochs: 10, device: cpu)
```

### Distributed Training

```esx
distribute(gpus: 2) {
    train(model, dataset, loss: ce, opt: adam(lr=0.001), epochs: 5)
}
```

### Autonomic Optimization

```esx
with autonomic {
    train(model, dataset, loss: ce, opt: adam(lr=0.001), epochs: 10)
}
```

### Memory Optimization

```esx
memory_broker(model, max_mem:8, strategy:zeRO)
quantize(model, bits:8, method:ptq)
checkpoint(model, segments:4)
```

### Custom Function

```esx
fn custom_loss(pred, target) {
    return mean((pred - target)^2)
}
```

## Error Handling

The parser provides detailed error messages with line and column information:

```
SyntaxError: Expected ')' at line 5, column 10
RuntimeError: Undefined variable 'x' at line 3, column 5
ValidationError: Invalid LoRA rank at line 7, column 15
```

## Implementation Notes

- The grammar is implemented using Boost.Spirit.Qi
- All expressions are left-associative
- Operator precedence follows standard mathematical conventions
- String literals support escape sequences
- Comments are ignored during parsing
- Whitespace is generally ignored except where significant
