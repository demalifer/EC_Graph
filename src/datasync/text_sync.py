import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from configuration.config import *
from utils import MysqlReader, Neo4jWriter
from ner.predict import Predictor

class TextSynchronizer():
    def __init__(self):
        self.reader = MysqlReader()
        self.writer = Neo4jWriter()
        # 定义实体的提取器，本质Predictor
        self.extractor = self._init_extractor()

    # 内部函数，初始化Predictor(extractor)
    def _init_extractor(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForTokenClassification.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / 'best_model.pt'))
        tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR / NER_DIR / 'best_model.pt'))
        return Predictor(model, tokenizer, device)

    # 同步Tag标签
    def sync_tag(self):
        # 1. 从MySQL提取商品描述信息
        sql = """
            select id, description
            from spu_info
        """
        spu_desc = self.reader.read(sql)

        # 2. 拆分spu_id和desc
        ids = [item['id'] for item in spu_desc]
        descs = [item['description'] for item in spu_desc]

        # 3. 提取所有商品数据的Tag列表
        tags_list = self.extractor.extract(descs)

        for id, tags in zip(ids, tags_list):
            print(id, tags)

        # 4. 构建Tag节点的属性(id, name)，以及SPU->Tag的关系(spu_id, tag_id)
        tag_properties = []
        relations = []
        for id, tags in zip(ids, tags_list):
            # 遍历当前SPU的每个标签
            for index, tag in enumerate(tags):
                # 构建Tag节点的属性
                tag_id = '-'.join([str(id), str(index)])
                propert = {'id': tag_id, 'name': tag}
                tag_properties.append(propert)
                # 构建关系
                relation = {'start_id': id, 'end_id': tag_id}
                relations.append(relation)
        # 5. 写入Neo4j
        self.writer.write_nodes('Tag', tag_properties)
        self.writer.write_relations('Have', 'SPU', 'Tag', relations)


if __name__ == "__main__":
    sync = TextSynchronizer()
    sync.sync_tag()