# TARDIS Knowledge Distillation

A lightweight implementation of knowledge distillation for classification tasks in PyTorch. This library enables transferring knowledge from a large pre-trained teacher model to a smaller, more efficient student model while maintaining comparable performance.

## Technical Description

Knowledge distillation is a model compression technique where a smaller student model learns not only from the ground truth labels but also from the soft targets (probability distributions) produced by a larger, more powerful teacher model. This approach allows the student model to benefit from the "dark knowledge" embedded in the teacher's outputs, often resulting in better performance than training on hard labels alone.

This implementation provides a simple yet effective framework for knowledge distillation training with customizable parameters such as temperature (controlling the softness of probability distributions) and teacher influence percentage.

## Dependencies

- Python 3.6+
- PyTorch 1.7+
- torchvision
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/TARDIS-KnowledgeDistillation.git
   cd TARDIS-KnowledgeDistillation
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

The main functionality is provided through the `knowledge_distillation_train` function:

```python
from knowledge_distilation import knowledge_distillation_train
import torch
import torch.nn as nn

# Initialize your teacher (pre-trained) and student models
teacher_model = your_pretrained_teacher_model
student_model = your_student_model

# Prepare your dataset and dataloader
trainloader = torch.utils.data.DataLoader(your_dataset, batch_size=32, shuffle=True)

# Setup optimizer
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

# Perform knowledge distillation training
epoch_loss = knowledge_distillation_train(
    teacher_model=teacher_model,
    student_model=student_model,
    trainloader=trainloader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    teacher_percentage=0.5,  # Balance between teacher signals and hard labels
    temperature=2.0,  # Controls softness of probability distributions
)
```

See `demo.py` for a complete example using a dummy dataset and a VGG19 teacher model.

### Parameters

| Parameter           | Type                           | Description                                                                    |
|--------------------|--------------------------------|--------------------------------------------------------------------------------|
| `teacher_model`    | `torch.nn.Module`             | The pre-trained teacher model used for knowledge distillation                  |
| `student_model`    | `torch.nn.Module`             | The student model that will learn from the teacher model                       |
| `trainloader`      | `torch.utils.data.DataLoader`  | The DataLoader providing the training data                                    |
| `criterion`        | `torch.nn.Module`             | The loss function used to compute the loss                                     |
| `optimizer`        | `torch.optim.Optimizer`        | The optimizer used to update the model parameters (e.g., `torch.optim.Adam`)  |
| `teacher_percentage`| `float`                       | The percentage of teacher model's output in loss calculation (default: 0.5)    |
| `temperature`      | `float`                       | The temperature parameter for softening the logits (default: 2)               |
| `device`           | `str` (optional)               | Device to run the models on ('cuda' or 'cpu'). If None, will use CUDA if available. |

### Returns

| Parameter    | Type    | Description                                  |
|-------------|---------|----------------------------------------------|
| `epoch_loss`| `float` | The average loss for the training epoch      |

## Features

- Easy-to-use knowledge distillation training interface
- Configurable temperature and teacher influence parameters
- Automatic device selection (GPU/CPU)
- Compatible with any PyTorch models and datasets
- Minimal dependencies

## Example Student Model

The repository includes a sample student model implementation in `models/student.py` designed for image classification tasks like CIFAR-10:

```python
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # Creates a lightweight CNN with 4 convolutional blocks and 3 fully connected layers
        # See the file for implementation details
        ...
```

## Acknowledgment

This work was partially supported by the "Trustworthy And Resilient Decentralised Intelligence For Edge Systems (TaRDIS)" Project, funded by EU HORIZON EUROPE program, under grant agreement No 101093006.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Feedback and Issues

Please report any bugs or feature requests through the [issue tracker](https://github.com/yourusername/TARDIS-KnowledgeDistillation/issues).