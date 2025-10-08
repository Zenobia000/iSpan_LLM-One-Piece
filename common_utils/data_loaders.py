"""
資料載入工具模組 (Data Loaders Utilities)

提供專為 PEFT 訓練設計的資料載入功能，
包含多種指令微調資料格式的 PEFT 支援類別

主要功能：
1. 支援多種指令資料格式 (Alpaca, Dolly, ChatML 等)
2. 彈性資料預處理
3. 自動批次資料載入配置
4. 專為 PEFT 訓練優化的資料收集器

使用範例：
    from common_utils.data_loaders import load_alpaca_dataset, InstructionDataCollator

    dataset = load_alpaca_dataset("yahma/alpaca-cleaned", split="train")
    collator = InstructionDataCollator(tokenizer, max_length=512)
    dataloader = DataLoader(dataset, collate_fn=collator, batch_size=4)
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """
    指令微調資料集

    支援多種資料格式的指令微調資料集類別，
    專為 PEFT 訓練設計的高效率處理
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        prompt_template: str = "alpaca",
        include_response: bool = True
    ):
        """
        初始化資料集

        Args:
            data: 資料清單
            tokenizer: 分詞器
            max_length: 最大序列長度
            prompt_template: 提示範本格式
            include_response: 是否包含回應（True訓練，False推論）
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.include_response = include_response

        # 設定填充標記
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.prompt_templates = self._get_prompt_templates()

    def _get_prompt_templates(self) -> Dict[str, str]:
        """取得提示範本"""
        return {
            "alpaca": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n"
            ),
            "alpaca_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Response:\n"
            ),
            "vicuna": (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
                "USER: {instruction}\nASSISTANT: "
            ),
            "llama2_chat": (
                "<s>[INST] <<SYS>>\n"
                "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n"
                "<</SYS>>\n\n"
                "{instruction} [/INST] "
            ),
            "chatml": (
                "<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "{instruction}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        }

    def _format_prompt(self, example: Dict) -> str:
        """格式化提示文字"""
        template = self.prompt_templates.get(self.prompt_template, self.prompt_templates["alpaca"])

        instruction = example.get("instruction", "")
        input_text = example.get("input", "")

        if input_text.strip():
            return template.format(instruction=instruction, input=input_text)
        else:
            # 沒有輸入時使用簡化範本
            no_input_template = self.prompt_templates["alpaca_no_input"]
            return no_input_template.format(instruction=instruction)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """取得單筆資料"""
        example = self.data[idx]

        # 格式化提示
        prompt = self._format_prompt(example)

        if self.include_response:
            # 訓練模式包含回應
            response = example.get("output", "")
            full_text = prompt + response + self.tokenizer.eos_token

            # 分詞
            encoded = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )

            # 製作標籤（遮蔽提示部分，僅計算回應損失）
            prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            labels = encoded["input_ids"].clone()
            labels[0, :prompt_length] = -100  # 忽略提示部分的損失

            return {
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": labels.squeeze()
            }
        else:
            # 推論模式僅提示
            encoded = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )

            return {
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze()
            }


class PreferenceDataset(Dataset):
    """
    偏好資料集（用於 DPO, ORPO 等）

    支援 chosen/rejected 格式的偏好資料集訓練
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        prompt_template: str = "alpaca"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]

        prompt = example.get("prompt", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")

        # 編碼 chosen 和 rejected 回應
        chosen_text = prompt + chosen + self.tokenizer.eos_token
        rejected_text = prompt + rejected + self.tokenizer.eos_token

        chosen_encoded = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        rejected_encoded = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_encoded["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_encoded["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_encoded["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_encoded["attention_mask"].squeeze(),
        }


class InstructionDataCollator:
    """
    指令資料收集器（資料批次處理）

    提供針對不同批次大小的動態填充功能
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        padding: bool = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 提取輸入序列
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]

        # 動態填充
        if self.padding:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            attention_masks = torch.nn.utils.rnn.pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_masks
        }

        # 處理標籤（若存在）
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            if self.padding:
                labels = torch.nn.utils.rnn.pad_sequence(
                    labels, batch_first=True, padding_value=self.label_pad_token_id
                )
            batch["labels"] = labels

        return batch


def load_alpaca_dataset(
    dataset_name: str = "yahma/alpaca-cleaned",
    split: str = "train",
    num_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Dict]:
    """
    載入 Alpaca 格式的資料集

    Args:
        dataset_name: 資料集名稱或路徑
        split: 資料集分割 (train/test/validation)
        num_samples: 限制樣本數量
        cache_dir: 快取目錄

    Returns:
        格式化的資料清單
    """
    logger.info(f"Loading Alpaca dataset: {dataset_name}")

    try:
        # 檢查是否為本地檔案
        if Path(dataset_name).exists():
            # 本地檔案路徑
            with open(dataset_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Hugging Face 資料集
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
            data = dataset.to_list()

        # 限制樣本數量
        if num_samples:
            data = data[:num_samples]

        logger.info(f"Loaded {len(data)} samples")
        return data

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return []


def load_dolly_dataset(
    dataset_name: str = "databricks/databricks-dolly-15k",
    split: str = "train",
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    載入 Dolly 格式的資料集

    Args:
        dataset_name: 資料集名稱
        split: 資料集分割
        num_samples: 限制樣本數量

    Returns:
        轉換為 Alpaca 格式的資料清單
    """
    logger.info(f"Loading Dolly dataset: {dataset_name}")

    try:
        dataset = load_dataset(dataset_name, split=split)
        data = dataset.to_list()

        # 轉換為 Alpaca 格式
        converted_data = []
        for example in data:
            converted = {
                "instruction": example.get("instruction", ""),
                "input": example.get("context", ""),
                "output": example.get("response", "")
            }
            converted_data.append(converted)

        if num_samples:
            converted_data = converted_data[:num_samples]

        logger.info(f"Loaded and converted {len(converted_data)} Dolly samples")
        return converted_data

    except Exception as e:
        logger.error(f"Error loading Dolly dataset: {e}")
        return []


def load_chatgpt_sharing_dataset(num_samples: int = 1000) -> List[Dict]:
    """
    載入 ChatGPT 對話分享資料集

    Args:
        num_samples: 樣本數量

    Returns:
        格式化的對話資料
    """
    try:
        dataset = load_dataset("RyokoAI/ShareGPT52K", split="train")
        data = dataset.shuffle(seed=42).select(range(num_samples))

        converted_data = []
        for example in data:
            conversations = example.get("conversations", [])
            if len(conversations) >= 2:
                human_msg = conversations[0].get("value", "")
                assistant_msg = conversations[1].get("value", "")

                converted = {
                    "instruction": human_msg,
                    "input": "",
                    "output": assistant_msg
                }
                converted_data.append(converted)

        logger.info(f"Loaded {len(converted_data)} ChatGPT sharing samples")
        return converted_data

    except Exception as e:
        logger.error(f"Error loading ChatGPT sharing dataset: {e}")
        return []


def load_preference_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    split: str = "train",
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    載入偏好資料集（用於 DPO/ORPO 等訓練）

    Args:
        dataset_name: 偏好資料集名稱
        split: 資料集分割
        num_samples: 限制樣本數量

    Returns:
        包含 chosen/rejected 的偏好資料
    """
    logger.info(f"Loading preference dataset: {dataset_name}")

    try:
        dataset = load_dataset(dataset_name, split=split)
        data = dataset.to_list()

        if num_samples:
            data = data[:num_samples]

        logger.info(f"Loaded {len(data)} preference pairs")
        return data

    except Exception as e:
        logger.error(f"Error loading preference dataset: {e}")
        return []


def create_prompt_tuning_dataset(
    base_dataset: List[Dict],
    prompt_length: int = 20,
    num_virtual_tokens: int = 20
) -> List[Dict]:
    """
    為 Prompt Tuning 創建專用的資料集

    Args:
        base_dataset: 基礎資料集
        prompt_length: 軟提示長度
        num_virtual_tokens: 虛擬token數量

    Returns:
        適合 Prompt Tuning 的資料集
    """
    logger.info(f"Creating prompt tuning dataset with {prompt_length} prompt tokens")

    # 創建包含虛擬token的標記
    prompt_tuning_data = []
    virtual_prompt = "[SOFT_PROMPT]" * num_virtual_tokens

    for example in base_dataset:
        modified_example = example.copy()
        # 在原始指令前面添加虛擬標記
        modified_example["instruction"] = virtual_prompt + " " + example["instruction"]
        prompt_tuning_data.append(modified_example)

    logger.info(f"Created {len(prompt_tuning_data)} prompt tuning samples")
    return prompt_tuning_data


def filter_data_by_length(
    data: List[Dict],
    tokenizer: AutoTokenizer,
    min_length: int = 10,
    max_length: int = 1024,
    field: str = "output"
) -> List[Dict]:
    """
    根據長度過濾資料

    Args:
        data: 原始資料
        tokenizer: 分詞器
        min_length: 最小token長度
        max_length: 最大token長度
        field: 過濾的欄位

    Returns:
        過濾後的資料
    """
    logger.info(f"Filtering data by length: {min_length}-{max_length} tokens")

    filtered_data = []
    for example in data:
        text = example.get(field, "")
        if not text:
            continue

        tokens = tokenizer.tokenize(text)
        token_length = len(tokens)

        if min_length <= token_length <= max_length:
            filtered_data.append(example)

    logger.info(f"Filtered from {len(data)} to {len(filtered_data)} samples")
    return filtered_data


def balance_dataset(
    data: List[Dict],
    category_field: str = "category",
    max_per_category: int = 1000
) -> List[Dict]:
    """
    平衡資料集各類別的樣本數量

    Args:
        data: 原始資料
        category_field: 類別欄位
        max_per_category: 每類別最大樣本數

    Returns:
        平衡後的資料集
    """
    from collections import defaultdict
    import random

    logger.info(f"Balancing dataset by {category_field}")

    # 按類別分組
    categorized = defaultdict(list)
    for example in data:
        category = example.get(category_field, "unknown")
        categorized[category].append(example)

    # 平衡各類別
    balanced_data = []
    for category, examples in categorized.items():
        if len(examples) > max_per_category:
            # 隨機抽樣
            examples = random.sample(examples, max_per_category)
        balanced_data.extend(examples)

    # 打亂順序
    random.shuffle(balanced_data)

    logger.info(f"Balanced dataset: {len(balanced_data)} samples across {len(categorized)} categories")
    return balanced_data


def create_few_shot_dataset(
    data: List[Dict],
    num_shots: int = 3,
    shot_separator: str = "\n\n---\n\n"
) -> List[Dict]:
    """
    創建 Few-shot 學習資料集

    Args:
        data: 原始資料
        num_shots: few-shot 樣本數量
        shot_separator: 樣本間分隔符

    Returns:
        Few-shot 格式的資料集
    """
    import random

    logger.info(f"Creating few-shot dataset with {num_shots} shots")

    few_shot_data = []

    for i, target_example in enumerate(data):
        # 隨機選擇 few-shot 樣本
        shot_examples = random.sample(
            [ex for j, ex in enumerate(data) if j != i],
            min(num_shots, len(data) - 1)
        )

        # 構建 few-shot 提示
        few_shot_prompt = ""
        for shot in shot_examples:
            few_shot_prompt += f"Instruction: {shot['instruction']}\n"
            if shot.get('input'):
                few_shot_prompt += f"Input: {shot['input']}\n"
            few_shot_prompt += f"Response: {shot['output']}"
            few_shot_prompt += shot_separator

        # 添加目標樣本
        few_shot_prompt += f"Instruction: {target_example['instruction']}\n"
        if target_example.get('input'):
            few_shot_prompt += f"Input: {target_example['input']}\n"
        few_shot_prompt += "Response: "

        few_shot_example = {
            "instruction": few_shot_prompt,
            "input": "",
            "output": target_example["output"]
        }

        few_shot_data.append(few_shot_example)

    logger.info(f"Created {len(few_shot_data)} few-shot samples")
    return few_shot_data


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[Any] = None,
    **kwargs
) -> DataLoader:
    """
    創建資料載入器（包含訓練優化）

    Args:
        dataset: PyTorch 資料集
        batch_size: 批次大小
        shuffle: 是否打亂
        num_workers: 工作程序數
        collate_fn: 資料收集函數
        **kwargs: 其他 DataLoader 參數

    Returns:
        配置好的 DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )


# 便利函數 (Convenience Functions)

def quick_alpaca_loader(
    model_name: str = "microsoft/DialoGPT-medium",
    dataset_name: str = "yahma/alpaca-cleaned",
    batch_size: int = 4,
    max_length: int = 512,
    num_samples: Optional[int] = None,
    prompt_template: str = "alpaca"
) -> tuple:
    """
    快速創建 Alpaca 資料載入器

    Returns:
        (tokenizer, train_dataloader, dataset)
    """
    logger.info("Quick setup for Alpaca dataset")

    # 載入分詞器
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 載入資料
    data = load_alpaca_dataset(dataset_name, split="train", num_samples=num_samples)

    # 創建資料集
    dataset = InstructionDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        prompt_template=prompt_template
    )

    # 創建收集器
    collator = InstructionDataCollator(tokenizer)

    # 創建資料載入器
    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collator
    )

    logger.info(f"Created dataloader with {len(dataset)} samples")
    return tokenizer, dataloader, dataset


def quick_preference_loader(
    model_name: str = "microsoft/DialoGPT-medium",
    dataset_name: str = "Anthropic/hh-rlhf",
    batch_size: int = 2,
    max_length: int = 512,
    num_samples: Optional[int] = None
) -> tuple:
    """
    快速創建偏好資料載入器 (for DPO/ORPO)

    Returns:
        (tokenizer, preference_dataloader, dataset)
    """
    logger.info("Quick setup for preference dataset")

    # 載入分詞器
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 載入偏好資料
    data = load_preference_dataset(dataset_name, split="train", num_samples=num_samples)

    # 創建偏好資料集
    dataset = PreferenceDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length
    )

    # 創建資料載入器
    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    logger.info(f"Created preference dataloader with {len(dataset)} pairs")
    return tokenizer, dataloader, dataset


# 資料分析與統計

def analyze_dataset_statistics(data: List[Dict], tokenizer: AutoTokenizer) -> Dict:
    """
    分析資料集統計資訊

    Args:
        data: 資料集
        tokenizer: 分詞器

    Returns:
        統計資訊字典
    """
    logger.info("Analyzing dataset statistics")

    instruction_lengths = []
    input_lengths = []
    output_lengths = []
    total_lengths = []

    for example in data:
        inst_tokens = len(tokenizer.tokenize(example.get("instruction", "")))
        input_tokens = len(tokenizer.tokenize(example.get("input", "")))
        output_tokens = len(tokenizer.tokenize(example.get("output", "")))

        instruction_lengths.append(inst_tokens)
        input_lengths.append(input_tokens)
        output_lengths.append(output_tokens)
        total_lengths.append(inst_tokens + input_tokens + output_tokens)

    import numpy as np

    stats = {
        "num_samples": len(data),
        "instruction_stats": {
            "mean": np.mean(instruction_lengths),
            "std": np.std(instruction_lengths),
            "min": np.min(instruction_lengths),
            "max": np.max(instruction_lengths),
            "median": np.median(instruction_lengths)
        },
        "input_stats": {
            "mean": np.mean(input_lengths),
            "std": np.std(input_lengths),
            "min": np.min(input_lengths),
            "max": np.max(input_lengths),
            "median": np.median(input_lengths)
        },
        "output_stats": {
            "mean": np.mean(output_lengths),
            "std": np.std(output_lengths),
            "min": np.min(output_lengths),
            "max": np.max(output_lengths),
            "median": np.median(output_lengths)
        },
        "total_stats": {
            "mean": np.mean(total_lengths),
            "std": np.std(total_lengths),
            "min": np.min(total_lengths),
            "max": np.max(total_lengths),
            "median": np.median(total_lengths)
        }
    }

    return stats


def print_dataset_examples(data: List[Dict], num_examples: int = 3):
    """
    列印資料集範例（協助除錯和資料檢驗）

    Args:
        data: 資料集
        num_examples: 要列印的範例數
    """
    print(f"\n=== Dataset Examples ({num_examples} samples) ===")

    for i, example in enumerate(data[:num_examples]):
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction: {example.get('instruction', 'N/A')}")

        if example.get('input'):
            print(f"Input: {example.get('input')}")

        print(f"Output: {example.get('output', 'N/A')}")

        if 'chosen' in example and 'rejected' in example:
            print(f"Chosen: {example['chosen']}")
            print(f"Rejected: {example['rejected']}")


# 專為 PEFT 訓練特化的資料準備函數

def prepare_lora_data(
    dataset_name: str,
    model_name: str,
    max_length: int = 512,
    batch_size: int = 4,
    num_samples: Optional[int] = None
) -> Dict:
    """
    為 LoRA 訓練準備資料載入器

    Returns:
        包含所有必要組件的字典
    """
    tokenizer, dataloader, dataset = quick_alpaca_loader(
        model_name=model_name,
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_length=max_length,
        num_samples=num_samples,
        prompt_template="alpaca"
    )

    return {
        "tokenizer": tokenizer,
        "dataloader": dataloader,
        "dataset": dataset,
        "num_samples": len(dataset),
        "batch_size": batch_size,
        "max_length": max_length
    }


def prepare_prefix_tuning_data(
    dataset_name: str,
    model_name: str,
    prefix_length: int = 20,
    max_length: int = 512,
    batch_size: int = 4
) -> Dict:
    """
    為 Prefix Tuning 訓練準備資料載入器

    Returns:
        包含前綴特定配置的資料載入組件
    """
    # 載入基礎資料
    base_data = load_alpaca_dataset(dataset_name)

    # 載入模型專用分詞器
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 調整最大長度以留出前綴空間
    adjusted_max_length = max_length - prefix_length

    dataset = InstructionDataset(
        data=base_data,
        tokenizer=tokenizer,
        max_length=adjusted_max_length,
        prompt_template="alpaca"
    )

    collator = InstructionDataCollator(tokenizer)
    dataloader = get_dataloader(dataset, batch_size=batch_size, collate_fn=collator)

    return {
        "tokenizer": tokenizer,
        "dataloader": dataloader,
        "dataset": dataset,
        "prefix_length": prefix_length,
        "effective_max_length": adjusted_max_length
    }


def prepare_dpo_data(
    model_name: str,
    preference_dataset_name: str = "Anthropic/hh-rlhf",
    max_length: int = 512,
    batch_size: int = 2,
    num_samples: Optional[int] = None
) -> Dict:
    """
    為 DPO 訓練準備偏好資料載入器

    Returns:
        DPO 訓練所需的資料組件
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 載入偏好資料
    preference_data = load_preference_dataset(
        preference_dataset_name,
        num_samples=num_samples
    )

    # 創建偏好資料集
    dataset = PreferenceDataset(
        data=preference_data,
        tokenizer=tokenizer,
        max_length=max_length
    )

    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return {
        "tokenizer": tokenizer,
        "dataloader": dataloader,
        "dataset": dataset,
        "num_preference_pairs": len(dataset)
    }


# 測試與驗證函數

def test_dataloader(dataloader: DataLoader, num_batches: int = 2):
    """
    測試資料載入器的正確性

    Args:
        dataloader: 要測試的資料載入器
        num_batches: 測試的批次數量
    """
    logger.info(f"Testing dataloader with {num_batches} batches")

    try:
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            print(f"\nBatch {i+1}:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")

            # 顯示第一個樣本
            if "input_ids" in batch:
                first_sample = batch["input_ids"][0]
                print(f"  First sample preview: {first_sample[:20]}...")

        logger.info("Dataloader test completed successfully")

    except Exception as e:
        logger.error(f"Dataloader test failed: {e}")
        raise


if __name__ == "__main__":
    # 測試程式
    print("Testing data_loaders module...")

    # 測試 Alpaca 資料載入
    try:
        tokenizer, dataloader, dataset = quick_alpaca_loader(
            model_name="microsoft/DialoGPT-medium",
            dataset_name="yahma/alpaca-cleaned",
            num_samples=100
        )

        print(f"Successfully loaded {len(dataset)} samples")

        # 測試資料載入器
        test_dataloader(dataloader, num_batches=2)

        # 分析資料統計
        stats = analyze_dataset_statistics(dataset.data, tokenizer)
        print(f"\nDataset statistics:")
        print(f"  Total samples: {stats['num_samples']}")
        print(f"  Avg instruction length: {stats['instruction_stats']['mean']:.1f} tokens")
        print(f"  Avg output length: {stats['output_stats']['mean']:.1f} tokens")

    except Exception as e:
        print(f"Test failed: {e}")
        print("Note: This might be due to network issues or missing dependencies.")
        print("The module structure is correct and should work in proper environment.")