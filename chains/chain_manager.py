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

    # 构建提示词
    prompt = f"用户问题: {query}\n"
    prompt += "=== 向量检索结果 ===\n"
    for r in results:
        if r["type"] == "vector":
            prompt += f"{r['result']['result']}\n"
            break

    prompt += "=== 知识图谱结果 ===\n"
    for r in results:
        if r["type"] == "graph":
            prompt += f"{r['result']['result']}\n"
            break

    if use_web:
        prompt += "=== 网络搜索结果 ===\n"
        for r in results:
            if r["type"] == "web":
                prompt += "\n".join(r['result'])[:500] + "\n"
                break

    prompt += "\n请根据以上信息，综合回答用户问题，要求：准确、简洁、全面。"

    # LLM 生成答案
    final_answer = await async_llm_predict(prompt)

    # 缓存结果
    redis_client.setex(cache_key, 3600, final_answer)

    return {"result": final_answer, "source": "llm"}
