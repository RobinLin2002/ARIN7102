import os
import torch
import streamlit as st
import ner_model as zwk
import pickle
import ollama
from transformers import BertTokenizer
import torch
import py2neo
import random
import re
import json
from tqdm import tqdm  
from py2neo import Graph


def load_model(cache_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open('tmp_data/tag2idx.npy', 'rb') as f:
        tag2idx = pickle.load(f)
    idx2tag = list(tag2idx)
    rule = zwk.rule_find()
    tfidf_r = zwk.tfidf_alignment()
    model_name = 'hfl/chinese-roberta-wwm-ext'
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = zwk.Bert_Model(model_name, hidden_size=128, tag_num=len(tag2idx), bi=True)
    bert_model.load_state_dict(torch.load(f'model/{cache_model}.pt', map_location=device))
    bert_model = bert_model.to(device)
    bert_model.eval()
    return bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device


def Intent_Recognition(query,choice):
    prompt = f"""
阅读下列提示，回答问题（问题在输入的最后）:
当你试图识别用户问题中的查询意图时，你需要仔细分析问题，并在16个预定义的查询类别中一一进行判断。对于每一个类别，思考用户的问题是否含有与该类别对应的意图。如果判断用户的问题符合某个特定类别，就将该类别加入到输出列表中。这样的方法要求你对每一个可能的查询意图进行系统性的考虑和评估，确保没有遗漏任何一个可能的分类。

**查询类别**
- "查询疾病简介"
- "查询疾病病因"
- "查询疾病预防措施"
- "查询疾病治疗周期"
- "查询治愈概率"
- "查询疾病易感人群"
- "查询疾病所需药品"
- "查询疾病宜吃食物"
- "查询疾病忌吃食物"
- "查询疾病所需检查项目"
- "查询疾病所属科目"
- "查询疾病的症状"
- "查询疾病的治疗方法"
- "查询疾病的并发疾病"
- "查询药品的生产商"

在处理用户的问题时，请按照以下步骤操作：
- 仔细阅读用户的问题。
- 对照上述查询类别列表，依次考虑每个类别是否与用户问题相关。
- 如果用户问题明确或隐含地包含了某个类别的查询意图，请将该类别的描述添加到输出列表中。
- 确保最终的输出列表包含了所有与用户问题相关的类别描述。

以下是一些含有隐晦性意图的例子，每个例子都采用了输入和输出格式，并包含了对你进行思维链形成的提示：
**示例1：**
输入："睡眠不好，这是为什么？"
输出：["查询疾病简介","查询疾病病因"]  # 这个问题隐含地询问了睡眠不好的病因
**示例2：**
输入："感冒了，怎么办才好？"
输出：["查询疾病简介","查询疾病所需药品", "查询疾病的治疗方法"]  # 用户可能既想知道应该吃哪些药品，也想了解治疗方法
**示例3：**
输入："跑步后膝盖痛，需要吃点什么？"
输出：["查询疾病简介","查询疾病宜吃食物", "查询疾病所需药品"]  # 这个问题可能既询问宜吃的食物，也可能在询问所需药品
**示例4：**
输入："我怎样才能避免冬天的流感和感冒？"
输出：["查询疾病简介","查询疾病预防措施"]  # 询问的是预防措施，但因为提到了两种疾病，这里隐含的是对共同预防措施的询问
**示例5：**
输入："头疼是什么原因，应该怎么办？"
输出：["查询疾病简介","查询疾病病因", "查询疾病的治疗方法"]  # 用户询问的是头疼的病因和治疗方法
**示例6：**
输入："如何知道自己是不是有艾滋病？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有艾滋病，一定一定要进行相关检查，这是根本性的！其次是查看疾病的病因，看看自己的行为是不是和病因重合。
**示例7：**
输入："我该怎么知道我自己是否得了21三体综合症呢？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有21三体综合症，一定一定要进行相关检查(比如染色体)，这是根本性的！其次是查看疾病的病因。
**示例8：**
输入："感冒了，怎么办？"
输出：["查询疾病简介","查询疾病的治疗方法","查询疾病所需药品","查询疾病所需检查项目","查询疾病宜吃食物"]  # 问怎么办，首选治疗方法。然后是要给用户推荐一些药，最后让他检查一下身体。同时，也推荐一下食物。
**示例9：**
输入："癌症会引发其他疾病吗？"
输出：["查询疾病简介","查询疾病的并发疾病","查询疾病简介"]  # 显然，用户问的是疾病并发疾病，随后可以给用户科普一下癌症简介。
**示例10：**
输入："葡萄糖浆的生产者是谁？葡萄糖浆是谁生产的？"
输出：["查询药品的生产商"]  # 显然，用户想要问药品的生产商
通过上述例子，我们希望你能够形成一套系统的思考过程，以准确识别出用户问题中的所有可能查询意图。请仔细分析用户的问题，考虑到其可能的多重含义，确保输出反映了所有相关的查询意图。

**注意：**
- 你的所有输出，都必须在这个范围内上述**查询类别**范围内，不可创造新的名词与类别！
- 参考上述5个示例：在输出查询意图对应的列表之后，请紧跟着用"#"号开始的注释，简短地解释为什么选择这些意图选项。注释应当直接跟在列表后面，形成一条连续的输出。
- 你的输出的类别数量不应该超过5，如果确实有很多个，请你输出最有可能的5个！同时，你的解释不宜过长，但是得富有条理性。

现在，你已经知道如何解决问题了，请你解决下面这个问题并将结果输出！
问题输入："{query}"
输出的时候请确保输出内容都在**查询类别**中出现过。确保输出类别个数**不要超过5个**！确保你的解释和合乎逻辑的！注意，如果用户询问了有关疾病的问题，一般都要先介绍一下疾病，也就是有"查询疾病简介"这个需求。
再次检查你的输出都包含在**查询类别**:"查询疾病简介"、"查询疾病病因"、"查询疾病预防措施"、"查询疾病治疗周期"、"查询治愈概率"、"查询疾病易感人群"、"查询疾病所需药品"、"查询疾病宜吃食物"、"查询疾病忌吃食物"、"查询疾病所需检查项目"、"查询疾病所属科目"、"查询疾病的症状"、"查询疾病的治疗方法"、"查询疾病的并发疾病"、"查询药品的生产商"。
"""
    rec_result = ollama.generate(model=choice, prompt=prompt)['response']
    # print(f'意图识别结果:{rec_result}')
    return rec_result
    # response, _ = glm_model.chat(glm_tokenizer, prompt, history=[])
    # return response


def add_shuxing_prompt(entity, shuxing, client):
    add_prompt = ""
    try:
        sql_q = "match (a:疾病{名称:'%s'}) return a.%s" % (entity, shuxing)
        res = client.run(sql_q).data()[0].values()
        add_prompt += f"<提示>"
        add_prompt += f"用户对{entity}可能有查询{shuxing}需求，知识库内容如下："
        if len(res) > 0:
            add_prompt += "".join(res)
        else:
            add_prompt += "图谱中无信息，查找失败。"
        add_prompt += f"</提示>"
    except:
        pass
    return add_prompt


def add_lianxi_prompt(entity, lianxi, target, client):
    add_prompt = ""
    try:
        sql_q = "match (a:疾病{名称:'%s'})-[r:%s]->(b:%s) return b.名称" % (entity, lianxi, target)
        res = client.run(sql_q).data()
        res = [list(data.values())[0] for data in res]
        add_prompt += f"<提示>"
        add_prompt += f"用户对{entity}可能有查询{lianxi}需求，知识库内容如下："
        if len(res) > 0:
            join_res = "、".join(res)
            add_prompt += join_res
        else:
            add_prompt += "图谱中无信息，查找失败。"
        add_prompt += f"</提示>"
    except:
        pass
    return add_prompt


def generate_prompt(response, query, options, client, bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag):
    entities = zwk.get_ner_result(bert_model, bert_tokenizer, query, rule, tfidf_r, device, idx2tag)
    yitu = []
    prompt = "<指令>你是一个医疗问答机器人，你需要根据给定的提示回答用户的问题。请注意，你的全部回答必须完全基于给定的提示，不可自由发挥”。</指令>"

    if '疾病症状' in entities and '疾病' not in entities:
        sql_q = "match (a:疾病)-[r:疾病的症状]->(b:疾病症状 {名称:'%s'}) return a.名称" % (entities['疾病症状'])
        res = list(client.run(sql_q).data()[0].values())
        if len(res) > 0:
            entities['疾病'] = random.choice(res)
            all_en = "、".join(res)
            prompt += f"<提示>用户有{entities['疾病症状']}的情况，知识库推测其可能是得了{all_en}。请注意这只是一个推测，你需要明确告知用户这一点。</提示>"

    pre_len = len(prompt)

    # 根据意图增加知识提示
    if "简介" in response:
        if '疾病' in entities:
            prompt += add_shuxing_prompt(entities['疾病'], '疾病简介', client)
            yitu.append('查询疾病简介')
    if "病因" in response:
        if '疾病' in entities:
            prompt += add_shuxing_prompt(entities['疾病'], '疾病病因', client)
            yitu.append('查询疾病病因')
    if "预防" in response:
        if '疾病' in entities:
            prompt += add_shuxing_prompt(entities['疾病'], '预防措施', client)
            yitu.append('查询疾病预防措施')
    if "治疗周期" in response:
        if '疾病' in entities:
            prompt += add_shuxing_prompt(entities['疾病'], '治疗周期', client)
            yitu.append('查询治疗周期')
    if "治愈概率" in response:
        if '疾病' in entities:
            prompt += add_shuxing_prompt(entities['疾病'], '治愈概率', client)
            yitu.append('查询治愈概率')
    if "易感人群" in response:
        if '疾病' in entities:
            prompt += add_shuxing_prompt(entities['疾病'], '疾病易感人群', client)
            yitu.append('查询疾病易感人群')
    if "药品" in response:
        if '疾病' in entities:
            prompt += add_lianxi_prompt(entities['疾病'], '疾病使用药品', '药品', client)
            yitu.append('查询疾病使用药品')
    if "宜吃食物" in response:
        if '疾病' in entities:
            prompt += add_lianxi_prompt(entities['疾病'], '疾病宜吃食物', '食物', client)
            yitu.append('查询疾病宜吃食物')
    if "忌吃食物" in response:
        if '疾病' in entities:
            prompt += add_lianxi_prompt(entities['疾病'], '疾病忌吃食物', '食物', client)
            yitu.append('查询疾病忌吃食物')
    if "检查项目" in response:
        if '疾病' in entities:
            prompt += add_lianxi_prompt(entities['疾病'], '疾病所需检查', '检查项目', client)
            yitu.append('查询疾病所需检查项目')
    if "所属科目" in response:
        if '疾病' in entities:
            prompt += add_lianxi_prompt(entities['疾病'], '疾病所属科目', '科目', client)
            yitu.append('查询疾病所属科目')
    if "症状" in response:
        if '疾病' in entities:
            prompt += add_lianxi_prompt(entities['疾病'], '疾病的症状', '疾病症状', client)
            yitu.append('查询疾病的症状')
    if "治疗" in response:
        if '疾病' in entities:
            prompt += add_lianxi_prompt(entities['疾病'], '治疗的方法', '治疗方法', client)
            yitu.append('查询疾病的治疗方法')
    if "并发" in response:
        if '疾病' in entities:
            prompt += add_lianxi_prompt(entities['疾病'], '疾病并发疾病', '疾病', client)
            yitu.append('查询疾病并发疾病')
    if "生产商" in response:
        try:
            sql_q = "match (a:药品商)-[r:生产]->(b:药品{名称:'%s'}) return a.名称" % (entities['药品'])
            res = client.run(sql_q).data()[0].values()
            prompt += f"<提示>用户对{entities['药品']}可能有查询药品生产商的需求，知识库内容如下："
            prompt += "".join(res) if res else "图谱中无信息，查找失败。"
            prompt += f"</提示>"
        except:
            pass

    if pre_len == len(prompt):
        prompt += "<提示>提示：知识库异常，没有相关信息！”。</提示>"

    # ========== 新增选择题要求 ==========
    option_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    prompt += f"""
<选择题要求>
你需要根据以上提示内容，在下面列出的四个选项中选择一个最合理的答案。
你必须只返回选项的字母（A/B/C/D），不要解释，不要重复题干内容。
</选择题要求>

题目：
{query}

选项：
{option_text}
"""
    return prompt, "、".join(yitu), entities

# ========== 评估主函数 ==========

def evaluate():
    os.makedirs('results', exist_ok=True)
    bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device = load_model('best_roberta_rnn_model_ent_aug')
    client= Graph("bolt://localhost:7687", auth=("neo4j", "20020501Lzy"))

    dataset = []
    with open('test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))

    dataset = dataset[:100]

    correct = 0
    total = len(dataset)
    predictions = []
    wrong_questions = []

    choice_model = 'deepseek-r1:8b'

    for idx, data in enumerate(tqdm(dataset, desc="正在评估选择题")):
        question = data['question']
        options = data['options']
        correct_answer_idx = data['answer_idx']

        # 意图识别
        intent_response = Intent_Recognition(question, choice_model)
        try:
            recognized_intents = eval(intent_response)
        except:
            recognized_intents = []

        # 构建Prompt
        prompt, _, _ = generate_prompt(recognized_intents, question, options, client, bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag)

        try:
            response = ollama.chat(model=choice_model, messages=[{'role': 'user', 'content': prompt}])
            model_raw_output = response['message']['content'].strip()
            model_answer = model_raw_output[0].upper() if model_raw_output else "?"
            if model_answer not in ['A', 'B', 'C', 'D']:
                model_answer = "?"
        except Exception as e:
            print(f"\n第{idx+1}题请求错误：{e}")
            model_answer = "?"
            model_raw_output = "错误"

        record = {
            'question': question,
            'options': options,
            'correct_answer': correct_answer_idx,
            'model_answer': model_answer,
            'prompt_used': prompt,
            'full_model_output': model_raw_output
        }
        predictions.append(record)

        if model_answer == correct_answer_idx:
            correct += 1
        else:
            wrong_questions.append(record)

    accuracy = correct / total
    print(f"\n最终准确率：{accuracy * 100:.2f}%")

    with open('results/mcq_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    with open('results/wrong_questions.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_questions, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    evaluate()
