# Parameter-Efficient Fine-Tuning of Qwen2.5-VL for Hazard-Aware VQA in Urban Environments

Hazard-aware Visual Question Answering (VQA) for autonomous driving using vision-language models.

This project investigates whether multimodal models can reason about potential hazards in urban environments by answering natural language questions about a driving scene.

---

## Project Summary

| Component | Description |
|-----------|-------------|
| Model | Qwen2.5-VL-3B |
| Task | Hazard detection via Visual Question Answering |
| Dataset | DRAMA-X |
| Method | Parameter-Efficient Fine-Tuning (QLoRA) |
| Frameworks | PyTorch, HuggingFace Transformers |

---

## Overview

Traditional autonomous driving systems rely on pipelines such as object detection, tracking, and trajectory prediction. These systems often lack a semantic understanding of intent.

This project reframes hazard detection as a multimodal reasoning problem.

The model receives:

- a visual driving scene
- a natural language question

and produces:

- hazard reasoning
- intent prediction
- suggested driving action

Example:

```
Question: What hazards are present?

Answer:
A cyclist is moving left toward the vehicle’s path.
Suggested Action: Slow down.
```


---

## Approach

The system fine-tunes Qwen2.5-VL-3B, a vision-language model capable of reasoning over images and video.

Training uses the DRAMA-X dataset, which provides intent labels for vulnerable road users such as pedestrians and cyclists.

Fine-tuning is performed using QLoRA, allowing efficient adaptation without updating the full model.

---

## Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- PEFT / QLoRA

## Dataset

- DRAMA-X – driving dataset with intent annotations for pedestrians and cyclists



