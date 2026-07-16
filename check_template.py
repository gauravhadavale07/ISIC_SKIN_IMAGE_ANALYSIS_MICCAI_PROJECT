from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("chaoyinshe/llava-med-v1.5-mistral-7b-hf")
if hasattr(processor.tokenizer, "chat_template") and processor.tokenizer.chat_template:
    print("Chat Template Exists:")
    print(processor.tokenizer.chat_template)
else:
    print("No chat template found.")
