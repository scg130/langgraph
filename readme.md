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
