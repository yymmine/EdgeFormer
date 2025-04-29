from transformers import OPTForCausalLM
import torch

def save_partitions():
    model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b")
    total_layers = len(model.model.decoder.layers)
    n = total_layers // 3
    
    # 保存前端部分
    front = {
        'embed_tokens': model.model.decoder.embed_tokens.state_dict(),
        'embed_positions': model.model.decoder.embed_positions.state_dict(),
        'layers': [layer.state_dict() for layer in model.model.decoder.layers[:n]]
    }
    torch.save(front, "opt_1.3b_front.pth")
    
    # 保存中间部分
    middle = {
        'layers': [layer.state_dict() for layer in model.model.decoder.layers[n:2*n]]
    }
    torch.save(middle, "opt_1.3b_middle.pth")
    
    # 保存后端部分
    back = {
        'layers': [layer.state_dict() for layer in model.model.decoder.layers[2*n:]],
        'final_layer_norm': model.model.decoder.final_layer_norm.state_dict(),
        'lm_head': model.lm_head.state_dict()
    }
    torch.save(back, "opt_1.3b_back.pth")