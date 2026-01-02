import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

"""
仅在basic RAG模式下可以使用personalized AI。重点：只有在用户与模型发生对话，并点击了Remember this按钮后，才会更新用户偏好文件！

1. 核心逻辑 (How it works)
这个功能构建了一个闭环的记忆系统，让 AI 能够“记住”用户的偏好、研究背景或特定的指令。
存储层 (Storage)：
基于 Client IP 作为唯一标识。
在本地 user_profiles/ 文件夹下存储纯文本文件（.txt）。
不需要向量数据库（Milvus），直接作为文本处理。
读取路径 (Read / Inject)：
用户提问时，系统检查侧边栏开关 Enable User Profile。
如果开启，读取对应的 .txt 文件内容。
将内容注入到 LLM 的提示词模板（Prompt Template）中的 {user_profile} 占位符。
LLM 看到这些上下文后，会根据你的偏好（如“使用 Python 代码”、“专注于 SFX”）生成回答。
写入路径 (Write / Update)：
手动模式：用户在侧边栏直接编辑文本框并保存，直接覆盖文件。
AI 自动模式：用户点击 "Remember This" 按钮。系统将“当前 Profile” + “刚才的 Q&A” 发送给 LLM（作为后台任务），让 LLM 总结出新的知识点并更新文件。
2. 如何使用 (User Guide)
第一步：开启功能
在左侧侧边栏（Sidebar），找到 "🧠 Personalized Memory" 区域，勾选 "Enable User Profile"。
第二步：预设偏好 (可选)
展开侧边栏的 "📝 View / Edit Profile"：
你可以在这里手动输入你的背景。
例子："我是做串行晶体学（SFX）的博士生，请多用物理公式解释原理，代码示例请使用 Python。"
点击 "💾 Save Profile"。系统会保存文件并刷新页面，AI 此刻起就知道你的身份了。
第三步：正常对话
在主界面输入问题。AI 在回答时会参考你的 Profile。
效果：如果你预设了“喜欢 Python”，AI 可能会主动提供代码，而不需要你每次都说“请给我代码”。
第三步：动态记忆
当你觉得某次对话非常有价值，或者你向 AI 纠正了一个错误后：
点击回答下方的 "🧠 Remember This" 按钮。
观察侧边栏：你会发现 "View / Edit Profile" 里的文本自动增加了关于这次对话的总结（例如：“用户对 Bragg 峰的积分算法感兴趣”）。
3. 场景示例
场景：你是一个初学者。
操作：在 Profile 写上 "Explain things simply, like I'm 5 years old."
结果：AI 的所有回答都会变得通俗易懂。
场景：你在进行特定的工程开发。
操作：在 Profile 写上 "Current project context: Processing EuXFEL data at 4.5 MHz rate."
结果：当你问“数据吞吐量是多少？”时，AI 会结合 4.5 MHz 这个参数来回答，而不是给出一个通用的数字。
"""

# 在当前目录下创建存储文件夹
PROFILE_DIR = "./user_profiles"
if not os.path.exists(PROFILE_DIR):
    os.makedirs(PROFILE_DIR)

def get_profile_path(user_id):
    # 将 IP 地址转换为合法文件名
    safe_id = str(user_id).replace(".", "_").replace(":", "_")
    return os.path.join(PROFILE_DIR, f"{safe_id}.txt")

def load_profile(user_id):
    """读取用户文档"""
    path = get_profile_path(user_id)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def overwrite_profile(user_id, content):
    """用户手动编辑保存，覆盖写入"""
    path = get_profile_path(user_id)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error saving profile: {e}")
        return False

def memorize_interaction(user_id, question, answer, llm):
    """使用 LLM 总结交互并更新文档"""
    path = get_profile_path(user_id)
    current_profile = load_profile(user_id)
    
    # 提示词：让 AI 提取偏好并合并到现有 Profile
    UPDATE_TEMPLATE = """
    You are the "Memory Manager" for ChatXFEL.
    
    TASK: Update the user's personalized research profile based on the IMPORTANT new interaction provided below.
    
    1. CURRENT PROFILE:
    {current_profile}
    
    2. NEW IMPORTANT INTERACTION:
    User Question: {question}
    AI Answer: {answer}
    
    3. INSTRUCTIONS:
    - Analyze the new interaction. What does it reveal about the user's research interests, technical level, or formatting preferences?
    - Merge these insights into the Current Profile.
    - If the Current Profile is empty, create a new one.
    - KEEP IT CONCISE: The profile should be a summary list of facts/preferences (e.g., "User focuses on serial crystallography", "User prefers Python examples").
    - Do NOT just copy-paste the conversation. Extract the *knowledge* about the user.
    
    4. UPDATED PROFILE (Text Only):
    """
    
    prompt = PromptTemplate.from_template(UPDATE_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    
    try:
        new_profile_content = chain.invoke({
            "current_profile": current_profile if current_profile else "No profile yet.",
            "question": question,
            "answer": answer
        })
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_profile_content)
        
        return True, new_profile_content
    except Exception as e:
        print(f"Error updating memory: {e}")
        return False, str(e)