#!/usr/bin/env python3
"""Scenario 2-3: image + defect-aware case2 description QLoRA training."""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CASE2_PROMPT = """You are analyzing an FDM 3D-printing image.

Identify the dataset defect label and describe the visible evidence that supports or contradicts it.
Use only evidence directly visible in the image. Do not invent causes or hidden conditions.

Return JSON in exactly this schema:
{
  "dataset_label": "Cracking|Layer_shifting|Off_platform|Stringing|Warping",
  "supporting_features": [
    {"feature_phrase": "short phrase", "evidence": "where and how it appears", "visibility": "clear|uncertain"}
  ],
  "contradicting_features": [],
  "label_consistency": "consistent|partially_consistent|inconsistent"
}

Rules:
- Name exactly one defect in dataset_label.
- Use one visible phenomenon per feature_phrase.
- Keep unsupported feature lists empty.
- Output JSON only.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    default_data = Path(os.environ.get("SCENARIO23_DATA", "./data"))
    parser.add_argument("--data-root", type=Path, default=default_data)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/scenario2-3"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--eval-steps", type=int, default=115)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.005)
    parser.add_argument("--expected-train", type=int, default=0,
                        help="0 disables an exact sample-count check")
    parser.add_argument("--expected-val", type=int, default=0)
    parser.add_argument("--resume", nargs="?", const="latest", default=None,
                        help="Resume from latest or from the supplied checkpoint path")
    return parser.parse_args()


def collect(root: Path, suffixes):
    if not root.is_dir():
        raise FileNotFoundError(f"Required directory not found: {root}")
    paths = {}
    duplicates = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in paths:
                duplicates.append(path.stem)
            paths[path.stem] = path
    if duplicates:
        raise RuntimeError(f"Duplicate sample IDs under {root}: {duplicates[:10]}")
    return paths


class Case2Dataset(Dataset):
    REQUIRED_KEYS = {
        "dataset_label", "supporting_features", "contradicting_features", "label_consistency"
    }

    def __init__(self, image_dir: Path, json_dir: Path, expected: int = 0):
        images = collect(image_dir, IMAGE_EXTENSIONS)
        labels = collect(json_dir, {".json"})
        missing = sorted(set(images) - set(labels))
        extra = sorted(set(labels) - set(images))
        if missing:
            raise RuntimeError(f"{len(missing)} images lack case2 JSON; examples: {missing[:10]}")
        if extra:
            print(f"Warning: ignoring {len(extra)} JSON files without images")
        self.samples = [(sample_id, images[sample_id], labels[sample_id]) for sample_id in sorted(images)]
        if not self.samples:
            raise RuntimeError(f"No matched image/JSON pairs in {image_dir} and {json_dir}")
        if expected and len(self.samples) != expected:
            raise RuntimeError(f"Expected {expected} samples, got {len(self.samples)}")

        for sample_id, _, label_path in self.samples:
            with label_path.open(encoding="utf-8") as handle:
                target = json.load(handle)
            missing_keys = self.REQUIRED_KEYS - set(target)
            if missing_keys:
                raise RuntimeError(f"{sample_id}: missing case2 keys {sorted(missing_keys)}")
        print(f"Loaded {len(self.samples)} pairs from {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id, image_path, label_path = self.samples[index]
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
        with label_path.open(encoding="utf-8") as handle:
            target = json.load(handle)
        return {"image": rgb, "target": json.dumps(target, ensure_ascii=False, indent=2)}


class VLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        encoded = []
        for example in examples:
            user = [{"role": "user", "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": CASE2_PROMPT},
            ]}]
            full = user + [{"role": "assistant", "content": [
                {"type": "text", "text": example["target"]}
            ]}]
            user_text = self.processor.apply_chat_template(
                user, tokenize=False, add_generation_prompt=True
            )
            full_text = self.processor.apply_chat_template(
                full, tokenize=False, add_generation_prompt=False
            )
            full_inputs = self.processor(
                text=[full_text], images=[example["image"]], return_tensors="pt"
            )
            user_inputs = self.processor(
                text=[user_text], images=[example["image"]], return_tensors="pt"
            )
            item = {key: value[0] for key, value in full_inputs.items()}
            item["labels"] = item["input_ids"].clone()
            item["labels"][: user_inputs["input_ids"].shape[1]] = -100
            encoded.append(item)

        batch = {}
        sequence_keys = {"input_ids", "attention_mask", "labels", "mm_token_type_ids"}
        max_len = max(item["input_ids"].shape[0] for item in encoded)
        for key in encoded[0]:
            values = [item[key] for item in encoded]
            if key in sequence_keys:
                pad_value = -100 if key == "labels" else (
                    self.processor.tokenizer.pad_token_id if key == "input_ids" else 0
                )
                values = [torch.nn.functional.pad(v, (0, max_len - v.shape[0]), value=pad_value)
                          for v in values]
            batch[key] = torch.stack(values)
        return batch


def main():
    args = parse_args()
    set_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = Case2Dataset(
        args.data_root / "train", args.data_root / "frontier_llm/gpt/case2", args.expected_train
    )
    val_dataset = Case2Dataset(
        args.data_root / "val", args.data_root / "frontier_llm/gpt/case2_val", args.expected_val
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, quantization_config=quantization, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = [name for name, _ in model.named_modules()
               if name.startswith("model.language_model.layers.")
               and name.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))]
    if not targets:
        raise RuntimeError("No language-model LoRA targets found; check transformers/model version")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=targets, task_type="CAUSAL_LM",
    ))
    if any(param.requires_grad and ("visual" in name.lower() or "vision" in name.lower())
           for name, param in model.named_parameters()):
        raise RuntimeError("Vision encoder unexpectedly has trainable parameters")
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4, warmup_ratio=0.05, lr_scheduler_type="cosine",
        bf16=True, fp16=False, gradient_checkpointing=True,
        logging_steps=10, eval_strategy="steps", save_strategy="steps",
        eval_steps=args.eval_steps, save_steps=args.eval_steps,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="tensorboard", logging_dir=str(args.output_dir / "tensorboard"),
        remove_unused_columns=False, dataloader_num_workers=2,
        optim="paged_adamw_8bit", seed=42,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=VLMDataCollator(processor),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )],
    )
    resume = None if args.resume is None else (True if args.resume == "latest" else args.resume)
    train_result = trainer.train(resume_from_checkpoint=resume)
    eval_result = trainer.evaluate()
    final_adapter = args.output_dir / "final_adapter"
    trainer.model.save_pretrained(final_adapter)
    processor.save_pretrained(final_adapter)
    summary = {
        "scenario": "2-3", "base_model": args.model_id,
        "max_epochs": args.epochs, "completed_epoch": trainer.state.epoch,
        "early_stopped": trainer.state.epoch < args.epochs,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_samples": len(train_dataset), "val_samples": len(val_dataset),
        "train_metrics": train_result.metrics, "eval_metrics": eval_result,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Training complete. Best adapter: {final_adapter}")


if __name__ == "__main__":
    main()

