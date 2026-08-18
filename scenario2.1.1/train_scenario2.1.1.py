import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# 0. Experiment configuration
# ============================================================

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

PROJECT_DIR = Path("/home/test-2/qwen_defect")

TRAIN_JSONL = (
    PROJECT_DIR
    / "datasets/final/train_labels(defect).jsonl"
)

VAL_JSONL = (
    PROJECT_DIR
    / "datasets/final/val_labels(defect).jsonl"
)

SCENARIO_DIR = (
    PROJECT_DIR
    / "result/scenario1_image_label"
)

CHECKPOINT_DIR = SCENARIO_DIR / "checkpoints"
TENSORBOARD_DIR = SCENARIO_DIR / "tensorboard"
FINAL_ADAPTER_DIR = SCENARIO_DIR / "final_adapter"
TEST_RESULT_DIR = SCENARIO_DIR / "test_predictions"
SUMMARY_PATH = SCENARIO_DIR / "training_summary.json"

for directory in [
    SCENARIO_DIR,
    CHECKPOINT_DIR,
    TENSORBOARD_DIR,
    TEST_RESULT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Fixed training hyperparameters
# ============================================================

NUM_EPOCHS = 3

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05

MAX_GRAD_NORM = 1.0

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

SEED = 42

LOGGING_STEPS = 10


# ============================================================
# 2. Minimal training instruction
# ============================================================

# 湲?Test Prompt???ъ슜?섏? ?딅뒗??
# Scenario 1? image -> defect label ?숈뒿留??섑뻾?쒕떎.

TRAIN_INSTRUCTION = (
    "Classify the defect in this FDM 3D-printing image. "
    "Return only the defect category."
)


# ============================================================
# 3. Label mapping for validation folders
# ============================================================

LABEL_MAP = {
    "Cracking": "cracking",
    "Layer_shifting": "layer_shifting",
    "Off_platform": "off_platform",
    "Stringing": "stringing",
    "Warping": "warping",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# 4. Train Dataset
#    train_labels.jsonl ?ъ슜
# ============================================================

class TrainJSONLDataset(Dataset):

    def __init__(self, jsonl_path):

        self.samples = []

        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Train JSONL not found: {jsonl_path}"
            )

        with open(
            jsonl_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(f, start=1):

                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)

                if "image" not in item:
                    raise ValueError(
                        f"Missing 'image' at line {line_number}"
                    )

                if "label" not in item:
                    raise ValueError(
                        f"Missing 'label' at line {line_number}"
                    )

                image_path = Path(item["image"])
                label = str(item["label"]).strip().lower()

                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Image not found at line "
                        f"{line_number}: {image_path}"
                    )

                self.samples.append(
                    {
                        "image_path": image_path,
                        "label": label,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# 5. Validation Dataset
#    val ?대뜑 ?대쫫?먯꽌 label ?먮룞 ?앹꽦
# ============================================================

class ValidationFolderDataset(Dataset):

    def __init__(self, root_dir):

        self.samples = []

        if not root_dir.exists():
            raise FileNotFoundError(
                f"Validation directory not found: {root_dir}"
            )

        for folder_name, label in LABEL_MAP.items():

            class_dir = root_dir / folder_name

            if not class_dir.exists():
                print(
                    f"[WARNING] Validation folder missing: "
                    f"{class_dir}"
                )
                continue

            image_paths = sorted(
                p
                for p in class_dir.rglob("*")
                if (
                    p.is_file()
                    and p.suffix.lower() in IMAGE_EXTENSIONS
                )
            )

            for image_path in image_paths:

                self.samples.append(
                    {
                        "image_path": image_path,
                        "label": label,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# 6. Load datasets
# ============================================================

train_dataset = TrainJSONLDataset(
    TRAIN_JSONL
)

val_dataset = TrainJSONLDataset(
    VAL_JSONL
)

print()
print("========================================")
print("Dataset information")
print("========================================")
print(f"Train samples      : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")
print("========================================")
print()


# ?곗씠??媛쒖닔 ?뺤씤
if len(train_dataset) != 1840:
    print(
        f"[WARNING] Expected 1840 train samples, "
        f"but found {len(train_dataset)}"
    )

if len(val_dataset) != 276:
    print(
        f"[WARNING] Expected 276 validation samples, "
        f"but found {len(val_dataset)}"
    )


# ============================================================
# 7. Processor
# ============================================================

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
)

# ============================================================
# 8. 4-bit NF4 configuration
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 9. Load Qwen3-VL-8B-Instruct
# ============================================================

print()
print(
    "Loading Qwen3-VL-8B-Instruct "
    "with 4-bit NF4..."
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

model.config.use_cache = False


# ============================================================
# 10. Prepare QLoRA
# ============================================================

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)


# ============================================================
# 11. LoRA configuration
#     LLM Decoder attention only
# ============================================================

lora_config = LoraConfig(

    r=LORA_R,

    lora_alpha=LORA_ALPHA,

    lora_dropout=LORA_DROPOUT,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    bias="none",

    task_type="CAUSAL_LM",
)


model = get_peft_model(
    model,
    lora_config,
)


# ============================================================
# 12. Vision part freeze
# ============================================================

# ?뱀떆 target module ?대쫫??vision ?곸뿭怨?寃뱀퀜??# Vision 履?LoRA源뚯? ?숈뒿?섏? ?딅룄濡??뺤떎?섍쾶 freeze.

for name, param in model.named_parameters():

    lower_name = name.lower()

    if (
        "visual" in lower_name
        or "vision" in lower_name
    ):
        param.requires_grad = False


# ============================================================
# 13. Verify trainable parameters
# ============================================================

print()
print("========================================")
print("Trainable parameters")
print("========================================")

model.print_trainable_parameters()

print()
print("Trainable parameter names:")

trainable_names = []

for name, param in model.named_parameters():

    if param.requires_grad:

        trainable_names.append(name)

        print(name)


# Safety check:
# Vision parameter媛 ?숈뒿 媛???곹깭硫?以묐떒

vision_trainable = [
    name
    for name in trainable_names
    if (
        "visual" in name.lower()
        or "vision" in name.lower()
    )
]

if vision_trainable:

    print()
    print("[ERROR]")
    print(
        "Vision parameters are unexpectedly trainable."
    )

    for name in vision_trainable:
        print(name)

    raise RuntimeError(
        "Vision Encoder must be frozen in Scenario 1."
    )


# ============================================================
# 14. Data Collator
#
# ?듭떖:
# Image + User Prompt       -> Loss X
# Assistant label          -> Loss O
# ============================================================

class Scenario1Collator:

    def __init__(self, processor):

        self.processor = processor

    def __call__(self, examples):

        # ?꾩옱 batch size = 1
        if len(examples) != 1:

            raise ValueError(
                "This collator is configured for "
                "per_device_batch_size=1."
            )

        sample = examples[0]

        image_path = sample["image_path"]
        label = sample["label"]

        image = Image.open(
            image_path
        ).convert("RGB")


        # ----------------------------------------------------
        # A. User prompt源뚯?留?議댁옱?섎뒗 ?낅젰
        #
        # ??湲몄씠瑜??댁슜?댁꽌 prompt 遺遺꾩쓣 -100?쇰줈 masking
        # ----------------------------------------------------

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": TRAIN_INSTRUCTION,
                    },
                ],
            }
        ]


        prompt_encoded = (
            self.processor.apply_chat_template(
                prompt_messages,

                tokenize=True,

                add_generation_prompt=True,

                return_dict=True,

                return_tensors="pt",
            )
        )


        prompt_length = (
            prompt_encoded["input_ids"].shape[1]
        )


        # ----------------------------------------------------
        # B. ?꾩껜 ?숈뒿 conversation
        #
        # User:
        # image + instruction
        #
        # Assistant:
        # cracking
        # ----------------------------------------------------

        full_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": TRAIN_INSTRUCTION,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": label,
                    }
                ],
            },
        ]


        encoded = (
            self.processor.apply_chat_template(
                full_messages,

                tokenize=True,

                add_generation_prompt=False,

                return_dict=True,

                return_tensors="pt",
            )
        )


        # ----------------------------------------------------
        # C. Labels ?앹꽦
        # ----------------------------------------------------

        labels = encoded[
            "input_ids"
        ].clone()


        # User prompt + image token 紐⑤몢 loss?먯꽌 ?쒖쇅
        labels[:, :prompt_length] = -100


        # padding??loss?먯꽌 ?쒖쇅
        if "attention_mask" in encoded:

            labels[
                encoded["attention_mask"] == 0
            ] = -100


        encoded["labels"] = labels


        return encoded


data_collator = Scenario1Collator(
    processor
)


# ============================================================
# 15. Training Arguments
# ============================================================

training_args = TrainingArguments(

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir=str(
        CHECKPOINT_DIR
    ),

    logging_dir=str(TENSORBOARD_DIR),


    # --------------------------------------------------------
    # Epoch / Batch
    # --------------------------------------------------------

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=(
        TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),


    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    warmup_steps=35,

    lr_scheduler_type="cosine",

    max_grad_norm=MAX_GRAD_NORM,

    optim="paged_adamw_8bit",


    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    bf16=True,

    gradient_checkpointing=True,


    # --------------------------------------------------------
    # TensorBoard logging
    # --------------------------------------------------------


    logging_strategy="steps",

    logging_steps=LOGGING_STEPS,

    logging_first_step=True,

    report_to="tensorboard",


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    eval_strategy="epoch",


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    save_strategy="epoch",

    save_total_limit=3,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,


    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,

    data_seed=SEED,


    # --------------------------------------------------------
    # Required for multimodal custom collator
    # --------------------------------------------------------

    remove_unused_columns=False,

    dataloader_num_workers=0,
)


# ============================================================
# 16. Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# ============================================================
# 17. Start training
# ============================================================

print()
print("========================================")
print("Scenario 1 Training Start")
print("========================================")

print(f"Model          : {MODEL_ID}")
print("Quantization   : 4-bit NF4")
print("LoRA target    : LLM Decoder only")
print(
    "LoRA modules   : "
    "q_proj, k_proj, v_proj, o_proj"
)
print(f"LoRA r         : {LORA_R}")
print(f"LoRA alpha     : {LORA_ALPHA}")
print(f"LoRA dropout   : {LORA_DROPOUT}")

print()

print(f"Train samples  : {len(train_dataset)}")
print(f"Val samples    : {len(val_dataset)}")
print(f"Epochs         : {NUM_EPOCHS}")

print(
    f"Batch size     : "
    f"{TRAIN_BATCH_SIZE}"
)

print(
    f"Grad accum     : "
    f"{GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"Effective batch: "
    f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"Learning rate  : "
    f"{LEARNING_RATE}"
)

print("========================================")
print()


start_time = time.time()


train_result = trainer.train()


end_time = time.time()


# ============================================================
# 18. Training time
# ============================================================

elapsed_seconds = (
    end_time - start_time
)

elapsed_minutes = (
    elapsed_seconds / 60
)

elapsed_hours = (
    elapsed_seconds / 3600
)


# ============================================================
# 19. Save best/final adapter
# ============================================================

trainer.save_model(
    str(FINAL_ADAPTER_DIR)
)

processor.save_pretrained(
    str(FINAL_ADAPTER_DIR)
)


# ============================================================
# 20. Final validation
# ============================================================

print()
print("Running final validation...")

eval_metrics = trainer.evaluate()


# ============================================================
# 21. Save summary
# ============================================================

summary = {

    "scenario": "scenario1_image_label",

    "model": MODEL_ID,

    "quantization": {
        "bits": 4,
        "type": "NF4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
    },

    "lora": {
        "target": "LLM Decoder only",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        "r": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
    },

    "dataset": {
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "train_jsonl": str(TRAIN_JSONL),
        "validation_jsonl": str(VAL_JSONL),
    },

    "training": {
        "epochs": NUM_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size":
            TRAIN_BATCH_SIZE
            * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": 35,
        "scheduler": "cosine",
        "optimizer": "paged_adamw_8bit",
        "max_grad_norm": MAX_GRAD_NORM,
        "seed": SEED,
    },

    "results": {
        "training_loss":
            float(train_result.training_loss),

        "elapsed_seconds":
            elapsed_seconds,

        "elapsed_minutes":
            elapsed_minutes,

        "elapsed_hours":
            elapsed_hours,

        "validation_metrics":
            {
                key: float(value)
                if isinstance(
                    value,
                    (int, float)
                )
                else value

                for key, value
                in eval_metrics.items()
            },
    },
}


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 22. Print final results
# ============================================================

print()
print("========================================")
print("Scenario 1 Training Finished")
print("========================================")

print(
    f"Training loss : "
    f"{train_result.training_loss:.6f}"
)

print(
    f"Training time : "
    f"{elapsed_hours:.2f} hours"
)

if "eval_loss" in eval_metrics:

    print(
        f"Validation loss: "
        f"{eval_metrics['eval_loss']:.6f}"
    )

print()
print(
    f"Final adapter:"
)
print(
    FINAL_ADAPTER_DIR
)

print()
print(
    f"Training summary:"
)
print(
    SUMMARY_PATH
)

print()
print(
    f"TensorBoard logs:"
)
print(
    TENSORBOARD_DIR
)

print("========================================")




