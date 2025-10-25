import asyncio
import os
from typing import Dict, List

import redis
import torch
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.chains.graph_qa.base import GraphQAChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from graphs.graph_manager import Neo4jManager
from search.search_manager import web_search

load_dotenv()


# ========== 基础配置 ==========
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] 当前设备: {device}")

# 假设你的模型都存放在 /Users/v_shemingdong/models/ 目录下
llm_model = (
    "Qwen/Qwen1.5-7B-Chat" if device == "cuda" else "Qwen/Qwen1.5-0.5B-Chat"
)
print(f"[INFO] 使用模型: {llm_model}")


# 初始化 LLM
tokenizer = AutoTokenizer.from_pretrained(
    llm_model, trust_remote_code=True, token=os.getenv("HUGGINGFACE_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(
    llm_model, trust_remote_code=True, token=os.getenv("HUGGINGFACE_TOKEN"))
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    truncation=True,
    temperature=0.7,
    device=0 if device == "cuda" else -1
)
llm = HuggingFacePipeline(pipeline=pipe)

# 异步封装 LLM predict


async def async_llm_predict(prompt: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, llm.invoke, prompt)

# ========== 向量数据库初始化 ==========


def get_chroma_store():
    embedding_model = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese")
    return Chroma(
        collection_name="docs",
        embedding_function=embedding_model,
        persist_directory="./chroma_store"
    )


chroma = get_chroma_store()
vector_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=chroma.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

# 异步包装向量检索


async def async_vector_query(query):
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, vector_chain.invoke, query)
    return {"type": "vector", "result": res}

# ========== 图谱查询 ==========
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# 创建 Networkx 图
graph = NetworkxEntityGraph()

neo4j_mgr = Neo4jManager(
    url=NEO4J_URL, user=NEO4J_USER, password=NEO4J_PASSWORD)
# 手动从 Neo4j 拉取节点和关系
nodes = neo4j_mgr.run_query("MATCH (n) RETURN n LIMIT 100")
edges = neo4j_mgr.run_query("MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100")

# 把数据加入 graph
for n in nodes:
    graph.add_node(n['n'].id, **dict(n['n']))

for e in edges:
    graph.add_edge(e['n'].id, e['m'].id, **dict(e['r']))

graph_chain = GraphQAChain.from_llm(llm=llm, graph=graph, verbose=False)

# 异步包装图谱查询


async def async_graph_query(query):
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, graph_chain.invoke, query)
    return {"type": "graph", "result": res}

# ========== 网络搜索 ==========


async def async_web_search(query):
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, web_search, query)
    return {"type": "web", "result": results}

# ========== 混合查询 ==========


async def hybrid_query(query: str, use_web: bool = True, force_refresh: bool = False):
    # 检查缓存
    cache_key = f"qa:{query}"
    if not force_refresh:
        cached = redis_client.get(cache_key)
        if cached:
            return {"result": cached.decode(), "source": "cache"}

    # 并行执行所有查询
    tasks = [async_vector_query(query), async_graph_query(query)]
    if use_web:
        tasks.append(async_web_search(query))

    results = await asyncio.gather(*tasks)

    # 构建更明确的提示词
    prompt = f"""你是一个专业的问答助手，需要根据提供的信息回答用户问题。
用户问题: {query}

信息来源：
"""

    # 向量检索结果
    vector_result = ""
    for r in results:
        if r["type"] == "vector" and r['result'].get('result'):
            vector_result = r['result']['result']
            break

    if vector_result:
        prompt += f"\n=== 向量检索结果 ===\n{vector_result}\n"

    # 知识图谱结果
    graph_result = ""
    for r in results:
        if r["type"] == "graph" and r['result'].get('result'):
            graph_result = r['result']['result']
            break

    if graph_result:
        prompt += f"=== 知识图谱结果 ===\n{graph_result}\n"

    # 网络搜索结果
    if use_web:
        web_result = ""
        for r in results:
            if r["type"] == "web" and r['result']:
                web_result = "\n".join(r['result'])[:500]
                break

        if web_result:
            prompt += f"=== 网络搜索结果 ===\n{web_result}\n"

    # 添加更明确的任务指示
    prompt += """
请根据以上信息，综合回答用户问题。要求：
1. 准确性：确保答案基于提供的信息，不要凭空猜测
2. 简洁性：用简洁明了的语言回答，避免冗长
3. 全面性：涵盖问题的关键点，不要遗漏重要信息
4. 格式要求：直接给出答案，不要添加"根据提供的信息"等额外说明
5. 如果没有足够信息回答，请明确说明"没有找到相关信息"""

    # LLM 生成答案
    final_answer = await async_llm_predict(prompt)

    # 添加答案后处理，过滤无用内容
    final_answer = process_llm_answer(final_answer)

    # 延长缓存时间到1小时
    redis_client.setex(cache_key, 6, final_answer)

    return {"result": final_answer, "source": "llm"}

# 添加答案后处理函数


def process_llm_answer(answer):
    """处理LLM生成的答案，过滤无用内容"""
    # 移除可能的前缀
    if isinstance(answer, str):
        # 去除多余的换行符和空格
        answer = answer.strip()

        # 移除常见的无用前缀
        prefixes = [
            "根据提供的信息",
            "基于提供的内容",
            "根据上述资料",
            "综合以上信息"
        ]

        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].lstrip("，。:：")
                break

        # 截断过长的答案（如果需要）
        max_length = 1000
        if len(answer) > max_length:
            answer = answer[:max_length] + "..."

    return answer
