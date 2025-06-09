# Fine-Tuning Vision Models with LoRA



This project demonstrates how to fine-tune vision models using the [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) technique on custom image datasets with the Hugging Face 🤗 Transformers and Datasets libraries.

---

## 🚀 Project Overview

- **Objective:** Efficiently fine-tune image classification models using LoRA to reduce training time and memory usage.
- **Datasets Used:**  
  - [Food-101](https://data.vision.ee.ethz.ch/cvl/food-101/) (subset for demo)
  - [Cats vs Dogs](https://www.tensorflow.org/datasets/community_catalog/huggingface/microsoft--cats_vs_dogs)
- **Key Features:**
  - Dynamic label mapping and preprocessing
  - Custom data collation for PyTorch
  - LoRA-adapted vision transformer models
  - Simple training and evaluation loop with Hugging Face Trainer


---

## 🛠️ Setup

1. **Install dependencies:**
   ```bash
   pip install torch torchvision transformers datasets pillow tqdm
   ```

2. **Download Datasets:**
   - Food-101: Download and extract the dataset as per instructions in your script.
   - Cats vs Dogs: Downloaded automatically via TensorFlow Datasets or Hugging Face Datasets.

---

## 🏗️ Example Workflow

### 1. Prepare Datasets

- Load image paths and labels for Food-101 and Cats vs Dogs.
- Use a dynamic label mapping utility to map class names to integer IDs.
- Preprocess images (e.g., resizing, normalization) and convert string labels to integer IDs.

### 2. Apply LoRA to Your Model

- Build your vision transformer model (e.g., ViT, Swin, etc.)
- Apply LoRA adapters to attention modules for efficient fine-tuning.

### 3. Training

- Use Hugging Face's `Trainer` with a custom data collator.
- Train and evaluate your model on the processed datasets.
- Save the fine-tuned model and report evaluation metrics.

---

## 📝 Example Code Snippet

```python
from datasets import Dataset
from transformers import Trainer
from lora_utils import build_lora_model
from data_utils import make_preprocess, create_label_mappings

# Prepare datasets
dataset = Dataset.from_list(samples)
label2id, id2label = create_label_mappings(dataset)
preprocess_fn = make_preprocess(label2id, preprocess_pipeline)
dataset = dataset.map(preprocess_fn, batched=True)

# Train
model = build_lora_model(label2id, id2label)
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collate,
)
trainer.train()
```

---

## 📊 Results

- LoRA enables faster training and reduced GPU memory consumption.
- Achieves competitive accuracy with a fraction of trainable parameters.

---

## 📚 References

- [LoRA: Low-Rank Adaptation of Large Language Models (arXiv)](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/index)
- [Hugging Face Datasets Docs](https://huggingface.co/docs/datasets/index)
- [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/food-101/)
- [Cats vs Dogs Dataset](https://www.tensorflow.org/datasets/community_catalog/huggingface/microsoft--cats_vs_dogs)

---
