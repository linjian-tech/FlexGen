import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


# 设置设备（自动选择 GPU 或 CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载数据
data = load_dataset("emozilla/pg19-test", split="test",cache_dir = "/media/hongzicong/volumn1/dataset_downloading")

# 模型名称（支持 HuggingFace 上的 Qwen3）
model_name = "Qwen/Qwen3-8B"
# 加载 Tokenizer 和 模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 使用 float16 节省显存
    device_map="auto",  # 自动分配到可用设备
    trust_remote_code=True,
    cache_dir="/media/hongzicong/volumn1/model_downloading"
)

prefill_length = 100000
seq_len = 10
for text in data["text"][:1]:
    encodings = tokenizer(text, return_tensors="pt")
    print(f"seq_len: {seq_len}")
    input_ids = encodings.input_ids[:, : prefill_length].to(device)
    input_length = input_ids.shape[1]
    with torch.no_grad():
        generate_ids = model.generate(
            input_ids,
            max_length=prefill_length + seq_len,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    print(f"prompt: {tokenizer.batch_decode(generate_ids[:, :input_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]}")
    print("############ output ###############")
    print(f"output: {tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]}")

# 推理函数
# def qwen_generate(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, do_sample=True):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,  # 使用 float16 节省显存
#         device_map="auto",  # 自动分配到可用设备
#         trust_remote_code=True,
#         cache_dir="/media/hongzicong/volumn1/model_downloading"
#     )
#
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             do_sample=do_sample,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id
#         )
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return generated_text
#
# # 测试输入
# prompt = "请告诉我一个关于人工智能的有趣事实。"
# response = qwen_generate(prompt)
#
# print("\n 模型输出：\n")
# print(response)