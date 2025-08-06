import torch
from collections import OrderedDict


def convert_sglang_to_megatron(sglang_weights, target_vocab_size=50257):
    megatron_weights = OrderedDict()
    
    # 1. 处理词表大小不匹配（裁剪或填充词嵌入）
    vocab_size, hidden_size = sglang_weights["transformer.wte.weight"].shape
    if vocab_size > target_vocab_size:
        # 裁剪多余的词嵌入
        megatron_weights["embedding.word_embeddings.weight"] = sglang_weights["transformer.wte.weight"][:target_vocab_size].clone()
    else:
        # 填充不足的词嵌入（用零初始化）
        pad_size = target_vocab_size - vocab_size
        padded_weight = torch.cat([
            sglang_weights["transformer.wte.weight"],
            torch.zeros(pad_size, hidden_size, dtype=torch.float16)
        ], dim=0)
        megatron_weights["embedding.word_embeddings.weight"] = padded_weight
    
    # 2. 位置嵌入（直接复制）
    megatron_weights["embedding.position_embeddings.weight"] = sglang_weights["transformer.wpe.weight"].float()
    
    # 3. 逐层转换
    num_layers = max([int(k.split('.')[2]) for k in sglang_weights.keys() if "h." in k]) + 1
    
    for layer_idx in range(num_layers):
        prefix = f"transformer.h.{layer_idx}"
        megatron_prefix = f"decoder.layers.{layer_idx}"
        
        # 3.1 LayerNorm 参数
        megatron_weights[f"{megatron_prefix}.input_layernorm.weight"] = sglang_weights[f"{prefix}.ln_1.weight"].float()
        megatron_weights[f"{megatron_prefix}.input_layernorm.bias"] = sglang_weights[f"{prefix}.ln_1.bias"].float()
        
        # 3.2 注意力层QKV处理
        qkv_weight = sglang_weights[f"{prefix}.attn.c_attn.weight"]  # [hidden_size, 3*hidden_size]
        qkv_bias = sglang_weights[f"{prefix}.attn.c_attn.bias"]      # [3*hidden_size]
        
        # 直接合并QKV（Megatron会在内部处理并行分块）
        megatron_weights[f"{megatron_prefix}.self_attention.linear_qkv.weight"] = qkv_weight
        megatron_weights[f"{megatron_prefix}.self_attention.linear_qkv.bias"] = qkv_bias
        
        # 3.3 注意力输出投影
        megatron_weights[f"{megatron_prefix}.self_attention.linear_proj.weight"] = sglang_weights[f"{prefix}.attn.c_proj.weight"]
        megatron_weights[f"{megatron_prefix}.self_attention.linear_proj.bias"] = sglang_weights[f"{prefix}.attn.c_proj.bias"]
        
        # 3.4 MLP层前置LayerNorm
        megatron_weights[f"{megatron_prefix}.pre_mlp_layernorm.weight"] = sglang_weights[f"{prefix}.ln_2.weight"].float()
        megatron_weights[f"{megatron_prefix}.pre_mlp_layernorm.bias"] = sglang_weights[f"{prefix}.ln_2.bias"].float()
        
        # 3.5 MLP层（关键修改：添加缺失的linear_fc2.bias）
        # fc1
        megatron_weights[f"{megatron_prefix}.mlp.linear_fc1.weight"] = sglang_weights[f"{prefix}.mlp.c_fc.weight"]
        megatron_weights[f"{megatron_prefix}.mlp.linear_fc1.bias"] = sglang_weights[f"{prefix}.mlp.c_fc.bias"]
        
        # fc2（权重直接复制，偏置初始化为零）
        megatron_weights[f"{megatron_prefix}.mlp.linear_fc2.weight"] = sglang_weights[f"{prefix}.mlp.c_proj.weight"]
        megatron_weights[f"{megatron_prefix}.mlp.linear_fc2.bias"] = sglang_weights[f"{prefix}.mlp.c_proj.bias"]
    
    # 4. 最终LayerNorm
    megatron_weights["decoder.final_layernorm.weight"] = sglang_weights["transformer.ln_f.weight"].float()
    megatron_weights["decoder.final_layernorm.bias"] = sglang_weights["transformer.ln_f.bias"].float()
    
    
    # 5. 移除所有_extra_state（Megatron会自动生成）
    return {k: v for k, v in megatron_weights.items() if not k.endswith("._extra_state")}
