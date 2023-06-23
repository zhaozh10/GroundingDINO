from transformers import GPT2Tokenizer

def get_tokenized_datasets(tokenizer, raw_datasets: list):

    # tokenized datasets will consist of the columns
    #   - mimic_image_file_path (str)
    #   - bbox_coordinates (List[List[int]])
    #   - bbox_labels (List[int])
    #   - bbox_phrases (List[str])
    #   - input_ids (List[List[int]])
    #   - attention_mask (List[List[int]])
    #   - bbox_phrase_exists (List[bool])
    #   - bbox_is_abnormal (List[bool])
    #
    #   val/test dataset will have additional column:
    #   - reference_report (str)

    def tokenize_function(example):
        phrases = example["bbox_phrases"]  # List[str]
        # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    tokenized_data=[raw_data.map(tokenize_function) for raw_data in raw_datasets]


    return tokenized_data


def get_tokenizer():
    checkpoint = "healx/gpt-2-pubmed-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer