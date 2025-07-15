from transformers import LlamaConfig


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
import os
os.environ['HF_HOME'] = '/media/hongzicong/volumn1/model_downloading'
# 设置模型名称
import os
print("HF_HOME:", os.environ.get('HF_HOME'))
model_name = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
config = LlamaConfig.from_pretrained(model_name)
print(config)
print(type(config))
print(config._attn_implementation)

# 加载分词器和模型

tokenizer = AutoTokenizer.from_pretrained(model_name)

# input_ids = tokenizer(prompts).input_ids
# print(input_ids)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
prompt = "Can you tell me a joke?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# with torch.no_grad():
#     outputs = model.generate(
#         inputs.input_ids,
#         max_new_tokens=100,
#         temperature=0.7,
#         top_p=0.9,
#         do_sample=True,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id
#     )

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=100,
        do_sample=False,            # 关闭采样
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response[len(prompt):].strip())

# 推理函数
# def generate_response(prompt, max_new_tokens=512):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs.input_ids,
#             max_new_tokens=max_new_tokens,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id
#         )
#
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response[len(prompt):].strip()


# 对话循环
# def chat_loop():
#     print("开始与 GradientAI Llama-3-8B 对话 (输入 '退出' 结束对话)\n")
#     while True:
#         user_input = input("用户: ")
#         if user_input.lower() in ["退出", "exit", "quit"]:
#             break
#
#         # 构造提示（你可以根据需要调整模板）
#         prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|start_header_id|>assistant<|end_header_id|>\n\n"
#
#         response = generate_response(prompt)
#         print(f"AI助手: {response}")
#
#
# if __name__ == "__main__":
#     chat_loop()
