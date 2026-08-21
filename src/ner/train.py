import os
from datasets import load_dataset, load_from_disk
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EvalPrediction,
    EarlyStoppingCallback,
)

from configuration.config import *
import time

run_log_dir = LOG_DIR / NER_DIR / time.strftime("%Y-%m-%d-%H-%M-%S")
run_log_dir.mkdir(parents=True, exist_ok=True)
os.environ["TENSORBOARD_LOGGING_DIR"] = str(run_log_dir)

# 1. 分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 标签映射
id2label = {id:label for id, label in enumerate(LABELS)}
label2id = {label:id for id, label in enumerate(LABELS)}

# 2. 模型
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)

# 3. 加载数据集
train_dataset = load_from_disk(PROCESSED_DATA_DIR / 'train')
valid_dataset = load_from_disk(PROCESSED_DATA_DIR / 'valid')

# 4. 数据整理器
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors='pt'
)

# 5. 训练参数
args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR / NER_DIR),
    logging_dir=str(run_log_dir),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,

    save_strategy='steps',
    save_steps=SAVE_STEPS,
    save_total_limit=3,

    fp16=True,

    logging_strategy='steps',
    logging_steps=SAVE_STEPS,
    report_to=['tensorboard'],

    eval_strategy='steps',
    eval_steps=SAVE_STEPS,

    metric_for_best_model='eval_overall_f1',
    greater_is_better=True,
    load_best_model_at_end=True,
)

# 6. 评估指标函数
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
        unpad_pred = [id2label[id] for id in unpad_pred]
        unpad_label = [id2label[id] for id in unpad_label]
        # 添加到列表
        unpad_labels.append(unpad_label)
        unpad_preds.append(unpad_pred)

    result = seqeval.compute(predictions=unpad_preds, references=unpad_labels)
    return result

# 7. 早停
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=20)

# 创建训练器
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    args=args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

# 训练
trainer.train()

# 模型保存
trainer.save_model(CHECKPOINT_DIR / NER_DIR / 'best_model.pt')
