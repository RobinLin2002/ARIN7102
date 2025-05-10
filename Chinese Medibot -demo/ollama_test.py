import ollama
import time

def test_ollama_generation():
    # 配置参数
    model_name = 'deepseek-r1:1.5b'
    prompt = "介绍一下人工智能的发展历程。"
    max_retries = 3
    timeout_seconds = 60
    
    print(f"\n{'='*50}")
    print(f"开始测试 Ollama 模型: {model_name}")
    print(f"输入提示词: {prompt}")
    print(f"{'='*50}\n")

    # 1. 检查模型是否已下载
    try:
        print("检查模型是否存在...")
        model_list = ollama.list()
        available_models = [m['name'] for m in model_list['models']]
        
        if model_name not in available_models:
            print(f"模型 {model_name} 未下载，正在拉取...")
            ollama.pull(model_name)
            print("模型下载完成！")
    except Exception as e:
        print(f"模型检查失败: {e}")
        return

    # 2. 尝试生成文本（带重试机制）
    for attempt in range(max_retries):
        try:
            print(f"\n尝试 #{attempt + 1} 生成响应...")
            start_time = time.time()
            
            # 带超时的生成请求
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'num_ctx': 2048
                }
            )
            
            # 验证响应
            if 'response' not in response:
                raise ValueError("响应中缺少 'response' 字段")
                
            elapsed_time = time.time() - start_time
            print(f"生成成功！耗时: {elapsed_time:.2f}秒")
            print(f"\n{'='*50}")
            print("生成结果:")
            print(response['response'])
            print(f"{'='*50}")
            return
            
        except ollama.ResponseError as e:
            print(f"API 响应错误: {e}")
        except TimeoutError:
            print(f"请求超时（>{timeout_seconds}秒）")
        except Exception as e:
            print(f"生成时出现意外错误: {e}")
        
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 5
            print(f"{wait_time}秒后重试...")
            time.sleep(wait_time)
    
    print("\n测试失败：达到最大重试次数")

if __name__ == "__main__":
    # 先检查Ollama服务是否运行
    try:
        print("检查Ollama服务状态...")
        ollama.list()  # 简单API调用测试连接
        print("Ollama服务已就绪！")
        test_ollama_generation()
    except Exception as e:
        print(f"无法连接Ollama服务: {e}")
        print("请确保已运行: ollama serve")