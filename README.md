# InferIQ

## Intelligent Inference & Reasoning Engine

InferIQ is a powerful application that combines deductive reasoning, Bayesian networks, probabilistic programming, and machine learning to provide accurate inferences with confidence scores.

## Features

- **Data & Rule Management**: Upload and manage data files and define reasoning rules
- **Inference & Validation**: Run inference on your data using multiple reasoning methods
- **Hyper-Parameter Tuning**: Optimize the weights of different reasoning methods
- **User Management**: Manage users with different roles and permissions
- **NLI**: Supports natural language to SWRL conversion

## Natural Language Interface Features

The system now supports natural language rule creation with:

- Natural language to SWRL conversion using fine-tuned LLMs
- Ontology-aware validation
- Interactive feedback for rule correction
- Bidirectional SWRL/Natural Language views

## Configuration

Edit the `.env` file to configure the application:

```
# Storage path for all file storage
INFERIQ_STORAGE=/path/to/your/storage
```

## Documentation

- [Login System](README_LOGIN.md): Detailed documentation of the login system.

## Using the Natural Language Interface:
- Load your ontology in the Ontology Management tab
- Switch to "Natural Language" input mode in Rule Management
- Describe your rule in plain English
- Click "Convert to SWRL" to generate formal rules
- Review and edit generated SWRL if needed
- Validate against your ontology before applying

## NLI to SWRL Validation Workflow
1. User Input: 
```
Senior customers get 10% discounts
```

2. LLM Processing:

```text
Customer(?c) ∧ hasAge(?c, ?a) ∧ greaterThanOrEqual(?a, 65) → applyDiscount(?c, 0.1)
```

3. Validation:

```text
Error: applyDiscount not in ontology
```

4. Feedback:

```text
Found issues:
- Undefined property: applyDiscount. Did you mean: hasDiscount, discountRate?
```

5. User Correction: 

```text
Use hasDiscount instead
```

6. Final Output:

```text
Customer(?c) ∧ hasAge(?c, ?a) ∧ swrlb:greaterThanOrEqual(?a, 65) → hasDiscount(?c, 0.1)
```
