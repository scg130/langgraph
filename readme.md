# 启动所有服务
docker-compose up --build -d

# 查看日志
docker logs -f langgraph_app

# 本地测试接口
curl -X POST -F 'query=中国和美国的贸易关系' http://localhost:8000/graph_qa

# 上传文件导入知识库
curl -X POST -F "file=@docs/测试文档.pdf" http://localhost:8000/ingest

# 获取流程图 DOT 格式
curl http://localhost:8000/flow_dot


OPENAI_API_KEY=
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
REDIS_URL=redis://localhost:6379/0
CHROMA_PERSIST_DIR=./chroma_store
HUGGINGFACE_TOKEN=