import os
import time
import streamlit as st
from transformers import AutoTokenizer, TextIteratorStreamer
from ipex_llm.transformers import AutoModelForCausalLM
import torch
from threading import Thread
from PIL import Image
import json

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "8"

# 定义可用的模型选项
model_options = {
    "模型1": "qwen2chat_int4",
    "模型2": "another_model_path"
}

# 加载和初始化模型
def load_model(model_path):
    model = AutoModelForCausalLM.load_low_bit(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def generate_response(messages, message_placeholder):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs=model_inputs.input_ids, max_new_tokens=512, streamer=streamer)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer

# Streamlit 应用部分
st.set_page_config(
    page_title="多模态大模型聊天应用",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("多模态大模型聊天应用")
st.write("上传图片、输入文本，与大模型互动并获取回复。")

# 多模型选择
selected_model = st.sidebar.selectbox("选择模型", list(model_options.keys()))
model_path = model_options[selected_model] 
model, tokenizer = load_model(model_path)

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入部分
if prompt := st.chat_input("你想说点什么?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response  = str()
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        streamer = generate_response(st.session_state.messages, message_placeholder)
        for text in streamer:
            response += text
            message_placeholder.markdown(response + "▌")

        message_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# 图片上传功能
uploaded_image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="上传的图片", use_column_width=True)
    # 可以在此处添加图像处理代码

# JSON解析功能
if st.button("解析JSON"):
    if st.session_state.messages:
        last_message = st.session_state.messages[-1]["content"]
        try:
            json_content = json.loads(last_message)
            st.json(json_content)
        except json.JSONDecodeError:
            st.write("最后一条消息不是有效的JSON格式")

# 聊天历史保存功能
if st.button("保存聊天历史"):
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=4)
    st.write("聊天历史已保存")

# 加载聊天历史功能
if st.button("加载聊天历史"):
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
        st.write("聊天历史已加载")
    else:
        st.write("没有找到保存的聊天历史文件")

# 美化页面
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# 添加公司Logo
st.sidebar.image("https://your-logo-url.com/logo.png", width=100)

# 添加联系和分享按钮
st.sidebar.markdown("""
    [![Star](https://img.shields.io/github/stars/yourusername/yourrepo.svg?logo=github&style=social)](https://github.com/yourusername/yourrepo)
    [![Follow](https://img.shields.io/twitter/follow/yourusername?style=social)](https://twitter.com/yourusername)
""")
