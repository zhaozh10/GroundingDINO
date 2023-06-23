from transformers import AutoTokenizer, GPT2LMHeadModel

checkpoint = "healx/gpt-2-pubmed-medium"
gpt_with_lm_head = GPT2LMHeadModel.from_pretrained(checkpoint)
lm_head = gpt_with_lm_head.lm_head
gpt = gpt_with_lm_head.transformer


gpt_children = list(gpt.children())

wte = gpt_children[0] 

