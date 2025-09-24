# LoRA

- Finetuned GPT-2 Base Model of 124M parameters on COLA dataset using LoRA technique.
- Number of Parameters are reduced by 99.5% (from 125M to 0.63M).
- Achieved an accuracy of 77.5%

  # Distillation

  - Trained a two Layer RNN network on COLA dataset, got accuracy of 65%.
  - Used the above Finetuned GPT model as teacher to train the student RNN model on COLA dataset, got 72% accuracy.