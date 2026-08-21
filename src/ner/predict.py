# 自定义预测器类
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from configuration.config import *

class Predictor:
    # 初始化
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    # 预测方法
    def predict(self, inputs: str|list[str]):
        # 如果是一条数据就转换成列表
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]

        # 1. 预分词，得到字符列表
        tokens_list = [list(input) for input in inputs]
        # 2. 用分词器ID化处理
        inputs_tensor = self.tokenizer(
            tokens_list,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # 3. 加载到设备
        inputs_tensor = {k:v.to(self.device) for k,v in inputs_tensor.items()}
        # 4. 前向传播，推理预测
        with torch.no_grad():
            outputs = self.model(**inputs_tensor)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        # 5. 将ID列表转换为BIO标签
        final_predictions = []
        for tokens, prediction in zip(tokens_list, predictions):
            # 截取预测输出中的真实长度
            prediction = prediction[1: len(tokens)+1]
            final_prediction = [self.model.config.id2label[int(id)] for id in prediction]
            final_predictions.append(final_prediction)

        if is_str:
            return final_predictions[0]
        return final_predictions

    def extract(self, inputs: str|list[str]):
        # 依旧，如果是一条数据，转换成列表处理。
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]
        # 得到预测标签列表
        predictions = self.predict(inputs)
        # 从当前列表抽取实体列表
        entities_list = []
        for input, labels in zip(inputs, predictions):
            entities = self._extract_entities(list(input), labels)
            entities_list.append(entities)
        if is_str:
            return entities_list[0]
        return entities_list

    def _extract_entities(self, tokens, labels):
        entities = []
        current_entity = ''
        for token, label in zip(tokens, labels):
            # 开始保存新的实体
            if label == 'B':
                if current_entity:
                    entities.append(current_entity)
                current_entity = token
            # 继续追加实体内容
            elif label == 'I':
                if current_entity:
                    current_entity += token
            # 将存在的实体添加到列表
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = ''
        if current_entity:
            entities.append(current_entity)
        return entities

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / 'best_model.pt'))
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / 'best_model.pt'))
    # 定义预测器
    predictor = Predictor(model, tokenizer, device)
    # 数据
    text = '麦德龙德国进口双芯多维叶黄素护眼营养软胶囊30粒x3盒眼干涩'
    # 预测
    result = predictor.predict(text)
    for token, label in zip(text, result):
        print(token, label)
    print('-'*50)
    # 抽取实体
    entities = predictor.extract(text)
    print(entities)

if __name__ == '__main__':
    predict()