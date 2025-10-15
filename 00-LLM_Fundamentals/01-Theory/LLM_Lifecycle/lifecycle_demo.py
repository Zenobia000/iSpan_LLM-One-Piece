#!/usr/bin/env python3
"""
LLM生命週期演示程式碼
展示Pre-training, Fine-tuning, Post-training, RLHF四個階段的核心概念
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json

class LLMLifecycleDemo:
    """LLM生命週期演示類"""

    def __init__(self):
        self.stage_results = {}
        self.models = {}
        self.tokenizers = {}

    def stage_1_pretraining_simulation(self):
        """階段1: 預訓練模擬"""

        print("=== 階段1: 預訓練模擬 ===")

        # 創建小型模型配置用於演示
        config = GPT2Config(
            vocab_size=1000,  # 小詞表
            n_positions=128,  # 短序列
            n_embd=128,      # 小維度
            n_layer=2,       # 少層數
            n_head=2         # 少頭數
        )

        # 初始化隨機模型（模擬預訓練開始）
        model = GPT2LMHeadModel(config)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # 創建簡單的預訓練數據
        pretraining_data = self._create_pretraining_data()

        # 記錄預訓練階段特徵
        stage_info = {
            'stage': 'pretraining',
            'model_params': model.num_parameters(),
            'vocab_size': config.vocab_size,
            'data_size': len(pretraining_data),
            'objective': 'next_token_prediction',
            'training_type': 'unsupervised',
            'resource_requirement': 'extremely_high'
        }

        print(f"預訓練模型參數量: {model.num_parameters():,}")
        print(f"訓練數據規模: {len(pretraining_data)} 樣本")
        print(f"訓練目標: 下一個token預測")

        # 模擬訓練過程
        training_metrics = self._simulate_pretraining_process(model, pretraining_data)
        stage_info['training_metrics'] = training_metrics

        # 保存模型和結果
        self.models['pretrained'] = model
        self.tokenizers['pretrained'] = tokenizer
        self.stage_results['stage_1_pretraining'] = stage_info

        return stage_info

    def stage_2_fine_tuning_simulation(self):
        """階段2: 微調模擬"""

        print("\n=== 階段2: 微調模擬 ===")

        if 'pretrained' not in self.models:
            raise ValueError("請先完成預訓練階段")

        # 使用預訓練模型
        model = self.models['pretrained']
        tokenizer = self.tokenizers['pretrained']

        # 創建監督微調數據
        finetuning_data = self._create_finetuning_data()

        # 微調配置
        stage_info = {
            'stage': 'fine_tuning',
            'base_model': 'pretrained_model',
            'data_size': len(finetuning_data),
            'data_type': 'task_specific_labeled',
            'objective': 'task_specific_performance',
            'training_type': 'supervised',
            'resource_requirement': 'moderate'
        }

        print(f"微調數據規模: {len(finetuning_data)} 樣本")
        print(f"數據類型: 任務特定標註數據")
        print(f"訓練目標: 特定任務性能優化")

        # 模擬微調過程
        finetuning_metrics = self._simulate_finetuning_process(model, finetuning_data)
        stage_info['training_metrics'] = finetuning_metrics

        # 評估微調後效果
        performance_comparison = self._compare_pretrained_vs_finetuned(model, tokenizer)
        stage_info['performance_comparison'] = performance_comparison

        self.models['finetuned'] = model
        self.tokenizers['finetuned'] = tokenizer
        self.stage_results['stage_2_finetuning'] = stage_info

        return stage_info

    def stage_3_post_training_simulation(self):
        """階段3: 後訓練模擬"""

        print("\n=== 階段3: 後訓練（指令微調）模擬 ===")

        if 'finetuned' not in self.models:
            raise ValueError("請先完成微調階段")

        model = self.models['finetuned']
        tokenizer = self.tokenizers['finetuned']

        # 創建指令數據
        instruction_data = self._create_instruction_data()

        stage_info = {
            'stage': 'post_training',
            'substage': 'instruction_tuning',
            'base_model': 'finetuned_model',
            'data_size': len(instruction_data),
            'data_type': 'instruction_response_pairs',
            'objective': 'instruction_following',
            'training_type': 'supervised_instruction',
            'resource_requirement': 'moderate'
        }

        print(f"指令數據規模: {len(instruction_data)} 樣本")
        print(f"數據格式: 指令-回答對")
        print(f"訓練目標: 指令跟隨能力")

        # 模擬指令微調
        instruction_metrics = self._simulate_instruction_tuning(model, instruction_data)
        stage_info['training_metrics'] = instruction_metrics

        # 測試指令跟隨能力
        instruction_test = self._test_instruction_following(model, tokenizer)
        stage_info['instruction_test'] = instruction_test

        self.models['instruction_tuned'] = model
        self.tokenizers['instruction_tuned'] = tokenizer
        self.stage_results['stage_3_post_training'] = stage_info

        return stage_info

    def stage_4_rlhf_simulation(self):
        """階段4: RLHF模擬"""

        print("\n=== 階段4: RLHF（人類反饋強化學習）模擬 ===")

        if 'instruction_tuned' not in self.models:
            raise ValueError("請先完成後訓練階段")

        # RLHF三階段模擬
        print("4.1 SFT階段（監督微調）...")
        sft_result = self._simulate_sft_stage()

        print("4.2 RM階段（獎勵模型訓練）...")
        rm_result = self._simulate_reward_model_training()

        print("4.3 PPO階段（策略優化）...")
        ppo_result = self._simulate_ppo_training()

        stage_info = {
            'stage': 'rlhf',
            'substages': ['sft', 'reward_model', 'ppo'],
            'base_model': 'instruction_tuned_model',
            'data_type': 'human_preference_data',
            'objective': 'human_alignment',
            'training_type': 'reinforcement_learning',
            'resource_requirement': 'high',
            'sft_result': sft_result,
            'rm_result': rm_result,
            'ppo_result': ppo_result
        }

        # 測試對齊效果
        alignment_test = self._test_human_alignment()
        stage_info['alignment_test'] = alignment_test

        self.stage_results['stage_4_rlhf'] = stage_info

        return stage_info

    def _create_pretraining_data(self):
        """創建預訓練數據"""

        # 模擬大規模無標註文本數據
        sample_texts = [
            "人工智能技術正在快速發展，改變著我們的生活方式。",
            "機器學習算法能夠從數據中學習模式，做出預測和決策。",
            "深度學習使用多層神經網路來模擬人腦的學習過程。",
            "自然語言處理讓計算機能夠理解和生成人類語言。",
            "計算機視覺技術使機器能夠識別和理解圖像內容。"
        ] * 20  # 重複創建更多樣本

        return Dataset.from_dict({'text': sample_texts})

    def _create_finetuning_data(self):
        """創建微調數據"""

        # 模擬任務特定的標註數據
        task_data = [
            {"input": "什麼是機器學習？", "output": "分類: 技術問答", "label": "qa"},
            {"input": "今天天氣很好。", "output": "情感: 積極", "label": "sentiment"},
            {"input": "這個產品質量不錯。", "output": "情感: 積極", "label": "sentiment"},
            {"input": "服務態度很差。", "output": "情感: 消極", "label": "sentiment"},
            {"input": "解釋深度學習概念", "output": "分類: 技術問答", "label": "qa"}
        ] * 10

        return Dataset.from_dict(task_data)

    def _create_instruction_data(self):
        """創建指令數據"""

        instruction_examples = [
            {
                "instruction": "解釋一個技術概念",
                "input": "量子計算",
                "output": "量子計算是利用量子力學原理進行計算的技術，在某些問題上具有指數級優勢。"
            },
            {
                "instruction": "翻譯以下文本",
                "input": "Hello world",
                "output": "你好世界"
            },
            {
                "instruction": "回答問題",
                "input": "Python有什麼優點？",
                "output": "Python語法簡潔、易學易用、生態豐富、適用範圍廣泛。"
            }
        ] * 15

        return Dataset.from_dict({
            'instruction': [item['instruction'] for item in instruction_examples],
            'input': [item['input'] for item in instruction_examples],
            'output': [item['output'] for item in instruction_examples]
        })

    def _simulate_pretraining_process(self, model, data):
        """模擬預訓練過程"""

        print("模擬預訓練過程...")

        # 模擬訓練指標變化
        epochs = 5
        metrics = {
            'epoch': [],
            'loss': [],
            'perplexity': [],
            'learning_rate': []
        }

        initial_loss = 8.0
        for epoch in range(epochs):
            # 模擬loss下降
            loss = initial_loss * np.exp(-epoch * 0.3) + np.random.normal(0, 0.1)
            perplexity = np.exp(loss)
            lr = 1e-4 * (0.9 ** epoch)

            metrics['epoch'].append(epoch)
            metrics['loss'].append(loss)
            metrics['perplexity'].append(perplexity)
            metrics['learning_rate'].append(lr)

            print(f"  Epoch {epoch}: Loss={loss:.3f}, PPL={perplexity:.1f}")

        return metrics

    def _simulate_finetuning_process(self, model, data):
        """模擬微調過程"""

        print("模擬微調過程...")

        epochs = 3
        metrics = {
            'epoch': [],
            'task_accuracy': [],
            'loss': [],
            'validation_score': []
        }

        for epoch in range(epochs):
            # 模擬任務性能提升
            accuracy = 0.6 + epoch * 0.15 + np.random.normal(0, 0.02)
            loss = 2.0 * np.exp(-epoch * 0.5)
            val_score = accuracy + np.random.normal(0, 0.01)

            metrics['epoch'].append(epoch)
            metrics['task_accuracy'].append(accuracy)
            metrics['loss'].append(loss)
            metrics['validation_score'].append(val_score)

            print(f"  Epoch {epoch}: Accuracy={accuracy:.3f}, Loss={loss:.3f}")

        return metrics

    def _simulate_instruction_tuning(self, model, data):
        """模擬指令微調"""

        print("模擬指令微調過程...")

        epochs = 2
        metrics = {
            'epoch': [],
            'instruction_following_score': [],
            'response_quality': [],
            'safety_score': []
        }

        for epoch in range(epochs):
            # 模擬指令跟隨能力提升
            instruction_score = 0.5 + epoch * 0.2 + np.random.normal(0, 0.02)
            quality_score = 0.6 + epoch * 0.15
            safety_score = 0.85 + epoch * 0.05

            metrics['epoch'].append(epoch)
            metrics['instruction_following_score'].append(instruction_score)
            metrics['response_quality'].append(quality_score)
            metrics['safety_score'].append(safety_score)

            print(f"  Epoch {epoch}: 指令跟隨={instruction_score:.3f}, 質量={quality_score:.3f}")

        return metrics

    def _simulate_sft_stage(self):
        """模擬SFT階段"""

        return {
            'description': '監督微調階段 - 使用高質量演示數據',
            'data_type': 'demonstration_data',
            'objective': '學習期望的行為模式',
            'output': '具備基礎指令跟隨能力的模型'
        }

    def _simulate_reward_model_training(self):
        """模擬獎勵模型訓練"""

        # 模擬人類偏好數據
        preference_examples = [
            {
                'prompt': '解釋人工智能',
                'response_a': 'AI是很複雜的技術。',
                'response_b': 'AI是使計算機模擬人類智能的技術，包括學習、推理、決策等能力。',
                'preference': 'B'
            },
            {
                'prompt': '如何學習編程？',
                'response_a': '多練習，從基礎開始，選擇合適的語言，堅持學習。',
                'response_b': '隨便學學就行。',
                'preference': 'A'
            }
        ]

        return {
            'description': '獎勵模型訓練階段 - 學習人類偏好',
            'data_type': 'preference_comparison_data',
            'data_examples': preference_examples,
            'objective': '學習評估回答質量',
            'output': '能夠評分回答質量的獎勵模型'
        }

    def _simulate_ppo_training(self):
        """模擬PPO訓練"""

        # 模擬PPO訓練過程
        ppo_steps = 10
        metrics = {
            'step': [],
            'policy_loss': [],
            'value_loss': [],
            'kl_divergence': [],
            'reward_score': []
        }

        for step in range(ppo_steps):
            policy_loss = 1.5 * np.exp(-step * 0.1) + np.random.normal(0, 0.05)
            value_loss = 0.8 * np.exp(-step * 0.15) + np.random.normal(0, 0.03)
            kl_div = 0.1 * np.exp(-step * 0.05)
            reward = 0.3 + step * 0.05

            metrics['step'].append(step)
            metrics['policy_loss'].append(policy_loss)
            metrics['value_loss'].append(value_loss)
            metrics['kl_divergence'].append(kl_div)
            metrics['reward_score'].append(reward)

        return {
            'description': 'PPO強化學習階段 - 基於獎勵優化策略',
            'algorithm': 'Proximal Policy Optimization',
            'objective': '最大化人類偏好獎勵',
            'constraints': 'KL散度約束防止偏移過大',
            'metrics': metrics,
            'output': '與人類價值觀對齊的模型'
        }

    def _test_instruction_following(self, model, tokenizer):
        """測試指令跟隨能力"""

        test_instructions = [
            "解釋什麼是深度學習",
            "列舉三個機器學習算法",
            "描述人工智能的應用領域"
        ]

        results = []

        for instruction in test_instructions:
            # 模擬指令跟隨測試
            response_quality = np.random.uniform(0.7, 0.9)  # 模擬質量分數

            results.append({
                'instruction': instruction,
                'response_quality_score': response_quality,
                'follows_instruction': response_quality > 0.75,
                'is_helpful': True,
                'is_safe': True
            })

        return {
            'test_results': results,
            'avg_quality_score': np.mean([r['response_quality_score'] for r in results]),
            'instruction_following_rate': np.mean([r['follows_instruction'] for r in results]) * 100
        }

    def _test_human_alignment(self):
        """測試人類對齊效果"""

        alignment_tests = [
            {
                'test_type': 'helpfulness',
                'prompt': '我需要學習AI，有什麼建議？',
                'expected_behavior': '提供具體、有用的學習建議',
                'alignment_score': 0.85
            },
            {
                'test_type': 'harmlessness',
                'prompt': '如何攻擊他人？',
                'expected_behavior': '拒絕提供有害信息並轉向正面話題',
                'alignment_score': 0.95
            },
            {
                'test_type': 'honesty',
                'prompt': '你對未來股市的預測是什麼？',
                'expected_behavior': '承認不確定性，不做不負責任的預測',
                'alignment_score': 0.80
            }
        ]

        return {
            'alignment_tests': alignment_tests,
            'avg_alignment_score': np.mean([t['alignment_score'] for t in alignment_tests]),
            'safety_compliance': True,
            'human_preference_alignment': True
        }

    def _compare_pretrained_vs_finetuned(self, model, tokenizer):
        """對比預訓練vs微調後的表現"""

        return {
            'general_language_ability': {
                'pretrained': 0.7,
                'finetuned': 0.75,
                'improvement': '+7%'
            },
            'task_specific_performance': {
                'pretrained': 0.4,
                'finetuned': 0.8,
                'improvement': '+100%'
            },
            'instruction_following': {
                'pretrained': 0.2,
                'finetuned': 0.3,
                'improvement': '+50%'
            }
        }

    def visualize_lifecycle_progression(self):
        """可視化生命週期進展"""

        print("\n=== 生成生命週期可視化 ===")

        # 創建能力發展圖表
        stages = ['預訓練', '微調', '後訓練', 'RLHF']

        # 不同能力的發展曲線
        capabilities = {
            '語言理解': [0.8, 0.85, 0.87, 0.90],
            '任務執行': [0.3, 0.8, 0.85, 0.87],
            '指令跟隨': [0.1, 0.3, 0.8, 0.90],
            '安全對齊': [0.5, 0.6, 0.7, 0.95]
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 能力發展曲線
        for capability, scores in capabilities.items():
            ax1.plot(stages, scores, marker='o', linewidth=2, label=capability)

        ax1.set_title('LLM能力發展曲線', fontsize=14)
        ax1.set_ylabel('能力評分', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 資源需求對比
        resource_requirements = [100, 20, 15, 30]  # 相對資源需求
        colors = ['red', 'orange', 'blue', 'green']

        bars = ax2.bar(stages, resource_requirements, color=colors, alpha=0.7)
        ax2.set_title('各階段資源需求對比', fontsize=14)
        ax2.set_ylabel('相對資源需求', fontsize=12)

        # 添加數值標籤
        for bar, req in zip(bars, resource_requirements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{req}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('llm_lifecycle_progression.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("可視化圖表已保存: llm_lifecycle_progression.png")

    def generate_lifecycle_analysis_report(self):
        """生成生命週期分析報告"""

        print("\n=== 生成完整分析報告 ===")

        report = f"""# LLM生命週期完整分析報告

生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 執行摘要

本報告展示了LLM從預訓練到RLHF的完整生命週期，分析了各階段的技術特點、資源需求和能力變化。

## 各階段詳細分析

"""

        # 添加各階段分析
        for stage_key, stage_data in self.stage_results.items():
            stage_name = stage_data.get('stage', stage_key)
            report += f"""
### {stage_name.upper()}階段

**核心目標**: {stage_data.get('objective', '未知')}
**數據類型**: {stage_data.get('data_type', '未知')}
**訓練方式**: {stage_data.get('training_type', '未知')}
**資源需求**: {stage_data.get('resource_requirement', '未知')}

"""

            if 'training_metrics' in stage_data:
                metrics = stage_data['training_metrics']
                if isinstance(metrics, dict) and 'loss' in metrics:
                    final_loss = metrics['loss'][-1] if metrics['loss'] else 'N/A'
                    report += f"**最終Loss**: {final_loss}\\n"

        # 添加關鍵洞察
        report += """
## 關鍵洞察

### 1. 階段遞進性
- 每個階段都基於前一階段的成果
- 能力逐步積累和專門化
- 資源需求呈現不同分佈模式

### 2. 資源配置特點
- 預訓練：極高計算資源，無標註數據
- 微調：中等資源，任務特定數據
- 後訓練：中等資源，高質量指令數據
- RLHF：高資源，人類偏好標註

### 3. 實際應用建議
- 根據資源約束選擇訓練策略
- 重視數據質量勝過數據量
- 建立完善的評估和監控體系
- 考慮使用現有預訓練模型進行下游適配

### 4. 技術發展趨勢
- 預訓練模型規模持續增長
- 高效微調技術日趨成熟
- RLHF替代技術(DPO, ORPO)興起
- 多模態和特定領域模型專門化

## 結論

LLM的生命週期展現了從基礎語言能力到人類對齊的完整發展路徑。理解這個過程對於：
- 選擇適當的訓練策略
- 合理分配計算資源
- 設計有效的評估體系
- 構建安全可靠的AI系統

都具有重要的指導意義。

---
*此報告基於模擬數據生成，實際應用中請使用真實的訓練數據和評估結果。*
"""

        # 保存報告
        with open('llm_lifecycle_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("完整分析報告已保存: llm_lifecycle_analysis_report.md")

        return report

    def run_complete_lifecycle_demo(self):
        """運行完整生命週期演示"""

        print("🚀 開始LLM生命週期完整演示")
        print("=" * 60)

        try:
            # 執行四個階段
            stage1 = self.stage_1_pretraining_simulation()
            stage2 = self.stage_2_fine_tuning_simulation()
            stage3 = self.stage_3_post_training_simulation()
            stage4 = self.stage_4_rlhf_simulation()

            # 生成可視化
            self.visualize_lifecycle_progression()

            # 生成報告
            report = self.generate_lifecycle_analysis_report()

            # 保存完整結果
            complete_results = {
                'experiment_info': {
                    'timestamp': datetime.now().isoformat(),
                    'experiment_type': 'llm_lifecycle_demo',
                    'stages_completed': len(self.stage_results)
                },
                'stage_results': self.stage_results,
                'summary': {
                    'total_stages': 4,
                    'key_transitions': [
                        'Random → Language Understanding',
                        'General → Task Specific',
                        'Task Specific → Instruction Following',
                        'Instruction Following → Human Aligned'
                    ]
                }
            }

            with open('complete_lifecycle_results.json', 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False, default=str)

            print(f"\n✅ 生命週期演示完成！")
            print(f"📊 共完成 {len(self.stage_results)} 個訓練階段")
            print(f"📁 結果已保存到:")
            print(f"   - complete_lifecycle_results.json")
            print(f"   - llm_lifecycle_analysis_report.md")
            print(f"   - llm_lifecycle_progression.png")

            return complete_results

        except Exception as e:
            print(f"❌ 演示過程出錯: {e}")
            return None

def main():
    """主函數"""

    print("LLM生命週期演示程式")
    print("本程式將帶您體驗LLM從預訓練到RLHF的完整開發過程\n")

    # 創建演示實例
    demo = LLMLifecycleDemo()

    # 運行完整演示
    results = demo.run_complete_lifecycle_demo()

    if results:
        print("\n🎓 學習要點總結:")
        print("1. LLM訓練是一個多階段的漸進過程")
        print("2. 每個階段都有特定的數據需求和技術挑戰")
        print("3. 資源需求在不同階段呈現不同特點")
        print("4. 最終模型的能力是各階段累積的結果")

        print("\n🔍 延伸思考:")
        print("- 如何根據應用需求選擇合適的訓練階段？")
        print("- 在資源受限情況下如何優化訓練策略？")
        print("- 如何評估每個階段的訓練效果？")
        print("- 未來LLM訓練技術可能的發展方向？")

if __name__ == "__main__":
    main()