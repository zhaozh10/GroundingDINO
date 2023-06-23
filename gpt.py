from transformers import AutoTokenizer, GPT2LMHeadModel
import torch

checkpoint = "healx/gpt-2-pubmed-medium"
gpt_with_lm_head = GPT2LMHeadModel.from_pretrained(checkpoint)
lm_head = gpt_with_lm_head.lm_head
gpt = gpt_with_lm_head.transformer


gpt_children = list(gpt.children())

wte = gpt_children[0] 

# 指定形状
shape = (3, 4)

# 生成随机矩阵
rand_matrix = torch.randint(low=10, high=101, size=shape, dtype=torch.long)


inputs_embeds = wte(rand_matrix)
print(f"inputs_embedding shape: {inputs_embeds.shape}")