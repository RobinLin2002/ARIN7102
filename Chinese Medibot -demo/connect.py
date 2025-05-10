from py2neo import Graph

try:
    # 连接配置（推荐使用 Bolt 协议）
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "20020501Lzy"))
    
    # 新方法获取 Neo4j 版本
    version_info = graph.run("CALL dbms.components()").data()
    neo4j_version = version_info[0]['versions'][0]
    
    print(f"连接成功！Neo4j 版本: {neo4j_version}")
    print("测试查询:", graph.run("MATCH (n) RETURN n LIMIT 1").data())
except Exception as e:
    print("连接失败:", e)
    print("请检查：")
    print("1. Neo4j 服务是否运行（任务管理器检查 'Neo4j' 服务）")
    print("2. 浏览器访问 http://localhost:7474 是否能打开")
    print("3. 密码是否正确（默认密码需要修改后才能使用）")