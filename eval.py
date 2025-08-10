from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import P
from transformers.generation import GenerationConfig
from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
tokenizer_path = "/root/LLM/glm3_tokenizer"
model_id = "/root/LLM/outputs/ckpt/ptm_tiny_llm_92m_epoch1/checkpoint-20000"
# model_id = "/root/LLM/outputs/ckpt/dpo_tiny_llm_92m_epoch1/checkpoint-374"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
# generation_config = GenerationConfig.from_pretrained(model_id, trust_remote_code=True)
generation_config = GenerationConfig()
sys_text = "你是由李小贱开发的个人助手。"
# user_text = "世界上最大的动物是什么？"
# user_text = "介绍一下刘德华。"
user_text = "介绍一下中国。"
input_txt = "\n".join([
    "<|system|>", sys_text.strip(), 
    "<|user|>", user_text.strip(), 
    "<|assistant|>"]).strip() + "\n"

generation_config.max_new_tokens = 200
generation_config.repetition_penalty=10.0
model_inputs = tokenizer(user_text, return_tensors="pt").to(model.device)
generated_ids = model.generate(model_inputs.input_ids, generation_config=generation_config)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)