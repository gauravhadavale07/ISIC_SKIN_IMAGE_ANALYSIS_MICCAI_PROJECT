import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    print("Loading model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    layer_idx = 15
    try:
        target_layer = model.language_model.model.layers[layer_idx]
    except AttributeError:
        target_layer = model.language_model.layers[layer_idx]

    # Create dummy steering vector and variables
    hidden_dim = model.language_model.config.hidden_size
    current_vec = torch.randn(hidden_dim, device=device, dtype=torch.float16)
    current_vec = current_vec / current_vec.norm()  # Unit norm
    current_alpha = 50.0  # Massive alpha
    text_norm = 6.0       # Typical norm
    
    call_count = 0
    def steering_hook(module, args, kwargs, output):
        nonlocal call_count
        call_count += 1
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output
        
        pre_norm = hidden_states.norm().item()
        
        seq_len = hidden_states.shape[1]
        v = current_vec.to(hidden_states.dtype).to(hidden_states.device)
        inject_length = 40
        
        if seq_len == 1:
            hidden_states = hidden_states + (current_alpha * text_norm) * v
        elif seq_len > inject_length:
            hidden_states = hidden_states.clone()
            hidden_states[:, -inject_length:, :] += (current_alpha * text_norm) * v
        else:
            hidden_states = hidden_states + (current_alpha * text_norm) * v
            
        post_norm = hidden_states.norm().item()
        print(f"[hook call {call_count}] alpha={current_alpha} vec_norm={v.norm().item():.4f} "
              f"seq_len={seq_len} pre={pre_norm:.4f} post={post_norm:.4f} delta={post_norm-pre_norm:.4f}")
              
        return (hidden_states,) + output[1:] if is_tuple else hidden_states

    print("Registering hook...")
    hook_handle = target_layer.register_forward_hook(steering_hook, with_kwargs=True)

    # Prepare dummy input (Image + Text)
    from PIL import Image
    dummy_img = Image.new('RGB', (224, 224), color = 'red')
    prompt = "USER: <image>\nAnalyze this clinical photograph. What is the diagnosis? Is the diagnosis MEL or NEV? Diagnosis: ASSISTANT:"
    
    print("Processing inputs...")
    inputs = processor(text=prompt, images=dummy_img, return_tensors="pt").to(device, torch.float16)
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
        
    print("Done. Hook called", call_count, "times.")

if __name__ == '__main__':
    main()
