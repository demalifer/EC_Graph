import pymysql
from neo4j import GraphDatabase
from pymysql.cursors import DictCursor

from configuration.config import MYSQL_CONFIG, NEO4J_CONFIG


# 读取 MySQL 工具类
class MysqlReader:
    # 初始化
    def __init__(self):
        self.connection = pymysql.connect(**MYSQL_CONFIG)
        self.cursor = self.connection.cursor(DictCursor)

    # 查询 MySQL，读取数据
    def read(self, sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    # 关闭连接
    def close(self):
        self.cursor.close()
        self.connection.close()

# 写入Neo4j的工具类
class Neo4jWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(**NEO4J_CONFIG)
    
    # 写入节点（批量，固定标签）
    def write_nodes(self, label: str, properties: list[dict]):
        cypher = f"""
                UNWIND $batch AS item
                MERGE (:{label} {{id: item.id, name: item.name}})
            """
        self.driver.execute_query(cypher, batch=properties)
    # 写入关系
    def write_relations(self, type: str, start_label, end_label, relations: list[dict]):
        cypher = f"""
                    UNWIND $batch AS item
                    MATCH (start:{start_label} {{id:item.start_id}}), (end:{end_label} {{id:item.end_id}})
                    MERGE (start)-[:{type}]->(end)
                """
        self.driver.execute_query(cypher, batch=relations)

if __name__ == '__main__':
    reader = MysqlReader()
    writer = Neo4jWriter()

    try:
        # 1. Category1
        # 1.1 读取 Category1
        sql = """
            select
            *
            from
            base_category1
        """
        category1 = reader.read(sql)
        print(category1)
        # 1.2 写入neo4j，标签Category1
        writer.write_nodes("category1", category1)

        # 2. Category2
        # 2.1 读取 Category2
        sql = """
              select
                  *
              from
                  base_category2
              """
        category2 = reader.read(sql)
        print(category2)
        # 2.2 写入neo4j，标签Category2
        writer.write_nodes("category2", category2)

        # 3. Category2 -Belong-> Category1
        # 3.1 读取base_category1 数据
        sql = """
            select id as start_id,
                category1_id as end_id
            from
            base_category2
        """
        relations = reader.read(sql)
        print(relations)
        # 3.2 写入neo4j，标签Belong
        writer.write_relations("Belong", start_label='category2', end_label='category1', relations=relations)
    finally:
        reader.close()