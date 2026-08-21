import evaluate
from datasets import load_from_disk
from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    EvalPrediction,
)

from configuration.config import *

# 1. 分词器
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR / NER_DIR / 'best_model.pt')

# 2. 模型
model = AutoModelForTokenClassification.from_pretrained(CHECKPOINT_DIR / NER_DIR / 'best_model.pt')

# 3. 加载数据集
test_dataset = load_from_disk(PROCESSED_DATA_DIR / 'test')
valid_dataset = load_from_disk(PROCESSED_DATA_DIR / 'valid')

# 4. 数据整理器
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors='pt'
)

# 5. 评估指标函数
seqeval = evaluate.load('seqeval')

def compute_metrics(prediction: EvalPrediction):
    # 提取模型的预测输出和真实标签
    logits = prediction.predictions
    preds = logits.argmax(axis=-1)
    labels = prediction.label_ids
    # 将标签ID转换为真实标签
    unpad_labels = []
    unpad_preds = []
    for pred, label in zip(preds, labels):
        # 去掉padding的id
        unpad_label = label[label != -100]
        unpad_pred = pred[label != -100]
        # 转换BIO标签
        unpad_pred = [model.config.id2label[id] for id in unpad_pred]
        unpad_label = [model.config.id2label[id] for id in unpad_label]
        # 添加到列表
        unpad_labels.append(unpad_label)
        unpad_preds.append(unpad_pred)

    result = seqeval.compute(predictions=unpad_preds, references=unpad_labels)
    return result

# 6. 定义训练器
trainer = Trainer(
    model=model,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7. 验证评估
result = trainer.evaluate()
print(result)
