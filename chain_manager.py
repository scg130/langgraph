import os
import asyncio
import torch
import redis
from typing import Dict, List
from googlesearch import search
from graph_manager import Neo4jManager
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.chains.graph_qa.base import GraphQAChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from dotenv import load_dotenv

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
tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True, token=os.getenv("HUGGINGFACE_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(llm_model, trust_remote_code=True, token=os.getenv("HUGGINGFACE_TOKEN"))
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    truncation=True,   # ⚡ 添加这一行
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
    embedding_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
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

neo4j_mgr = Neo4jManager(url=NEO4J_URL, user=NEO4J_USER, password=NEO4J_PASSWORD)
# 手动从 Neo4j 拉取节点和关系
nodes = neo4j_mgr.run_query("MATCH (n) RETURN n LIMIT 100")  # 自定义查询
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
def web_search(question: str, max_results: int = 3) -> List[Dict[str, str]]:
    try:
        results = []
        for url in search(question, num_results=max_results, advanced=True):
            results.append({"title": url.title,  "body": url.description})
        return results
    except Exception as e:
        return [f"网络搜索出错：{str(e)}"]

async def async_web_search(query):
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, web_search, query)
    return {"type": "web", "result": res}

# ========== 问题分类 ==========
async def classify_question(question: str) -> str:
    """判断问题类型：关系型用 'graph'，否则用 'vector'"""
    prompt = f"判断问题类型：如果问题涉及人物、地点、关系，用 'graph'，否则用 'vector'。\n问题：{question}"
    resp = await async_llm_predict(prompt)
    return "graph" if "graph" in resp.lower() else "vector"

# ========== 混合问答主逻辑 ==========
async def hybrid_query(query: str, use_web: bool = True, force_refresh: bool = False):
    cache_key = f"qa_cache:{query}"
    if not force_refresh:
        cached = redis_client.get(cache_key)
        if cached:
            return {"source": "cache", "answer": cached.decode("utf-8")}

    # 并发执行三种查询
    tasks = [
        asyncio.create_task(async_vector_query(query)),
        asyncio.create_task(async_graph_query(query))
    ]
    if use_web:
        tasks.append(asyncio.create_task(async_web_search(query)))

    done, _ = await asyncio.wait(tasks)
    vector_res, graph_res, web_res = None, None, None

    for t in done:
        r = t.result()
        if r.get("type") == "vector":
            vector_res = r["result"]
        elif r.get("type") == "graph":
            graph_res = r["result"]
        elif r.get("type") == "web":
            web_res = r["result"]

    # 由模型智能综合回答
    label = await classify_question(query)
    summary_prompt = f"""
请根据以下信息整合回答用户问题：
问题：{query}
向量结果：{vector_res}
图谱结果：{graph_res}
网络结果：{web_res}
请输出简洁准确的中文答案。
"""
    final_answer = await async_llm_predict(summary_prompt)
    answer = final_answer.strip()

    # 缓存答案
    redis_client.set(cache_key, answer, ex= 6)
    return {
        "source": label,
        "answer": answer,
        "parts": {"vector": vector_res, "graph": graph_res, "web": web_res}
    }


