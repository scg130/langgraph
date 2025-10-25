import os

from fastapi import FastAPI, Form, UploadFile

from chains.chain_manager import hybrid_query
from graphs.langgraph_flow import FlowBuilder
from vectorstore.ingest import ingest_file

os.makedirs("./uploads", exist_ok=True)
app = FastAPI(title="LangGraph Hybrid QA Service", version="1.0")


@app.post("/graph_qa")
async def graph_qa(query: str = Form(...), use_web: bool = True):
    """混合问答接口：自动判断使用图谱 / 向量 / 联网"""
    result = await hybrid_query(query, use_web=use_web, force_refresh=True)
    return result


@app.post("/ingest")
async def ingest_doc(file: UploadFile):
    """导入文档到 Chroma 向量数据库"""
    path = f"./uploads/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    count = ingest_file(path)
    return {"status": "ok", "chunks": count, "filename": file.filename}


@app.get("/flow_dot")
def flow_dot():
    """返回 LangGraph 流程图 DOT 格式"""
    fb = FlowBuilder()
    fb.add_node("start", "用户问题")
    fb.add_node("vector", "向量检索")
    fb.add_node("graph", "知识图谱")
    fb.add_node("web", "联网搜索")
    fb.add_node("merge", "结果融合")
    fb.add_node("end", "输出答案")

    fb.add_edge("start", "vector")
    fb.add_edge("start", "graph")
    fb.add_edge("start", "web")
    fb.add_edge("vector", "merge")
    fb.add_edge("graph", "merge")
    fb.add_edge("web", "merge")
    fb.add_edge("merge", "end")

    return {"dot": fb.to_dot()}


@app.get("/")
def index():
    return {"message": "LangGraph Hybrid QA Service 已启动 🚀"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
