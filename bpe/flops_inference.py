from collections import OrderedDict


# 用于计算一个Transformer模型中不同部分的浮点运算次数（FLOPs）
def flops(block_size, vocab_size, n_layer, n_head, n_embd):
    # we only count Weight FLOPs, all other layers (LayerNorm, Softmax, etc) are effectively irrelevant
    # we count actual FLOPs, not MACs. Hence 2* all over the place
    # basically for any matrix multiply (BxC) @ (CxD) -> (BxD) flops are 2*B*C*D
    # 只计算权重相关的FLOPs，忽略其他层如LayerNorm和Softmax；计算的是实际的FLOPs而不是乘加次数（MACs）

    out = OrderedDict()
    head_size = n_embd // n_head

    # attention blocks
    # 1) the projection to key, query, values
    # 没有计算B这个维度
    out['attention/kqv'] = 2 * block_size * (n_embd * 3 * n_embd)
    # 2) calculating the attention scores (q*v) (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    out['attention/scores'] = 2 * block_size * block_size * n_embd
    # 3) the reduction of the values (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    out['attention/reduce'] = 2 * n_head * (block_size * block_size * head_size)
    # 4) the final linear projection
    out['attention/proj'] = 2 * block_size * (n_embd * n_embd)
    out['attention'] = sum(out['attention/' + k] for k in ['kqv', 'scores', 'reduce', 'proj'])

    # MLP blocks
    ffw_size = 4 * n_embd  # feed forward size
    out['mlp/ffw1'] = 2 * block_size * (n_embd * ffw_size)
    out['mlp/ffw2'] = 2 * block_size * (ffw_size * n_embd)
    out['mlp'] = out['mlp/ffw1'] + out['mlp/ffw2']

    # the transformer and the rest of it
    out['block'] = out['attention'] + out['mlp']
    out['transformer'] = n_layer * out['block']
    out['dense'] = 2 * block_size * (n_embd * vocab_size)

    # forward,backward,total
    out['forward_total'] = out['transformer'] + out['dense']
    out['backward_total'] = 2 * out['forward_total']  # use common estimate of bwd = 2*fwd
    out['total'] = out['forward_total'] + out['backward_total']
    print(f"block_size:{block_size:8d} vocab_size:{vocab_size:6d} n_layer:{n_layer:3d} n_head:{n_head:3d} n_embd:{n_embd:5d} "
          f"flops_total:{out['forward_total']  / 1e12:8.6f} flops_per:{out['forward_total']  / block_size/ 1e12:4.6f} "
          f"flops_total_price:{out['forward_total'] * 430.272 / 989 / 0.3 / 3600 / 24 / 1e12 :8.6f} "
          f"flops_per_price:{out['forward_total']  / block_size * 430.272 / 989 / 0.3 / 3600 / 24 / 1e12:4.8f}")
    return out



# nanogpt
f = flops(1024, 50257, 12, 12, 768)
# gpt3
f = flops(1024 * 2, 50257, 96, 96, 12288)
# gpt3.5
f = flops(1024 * 4, 100276, 96, 96, 12288)
f = flops(1024 * 8, 100276, 96, 96, 12288)
f = flops(1024 * 16, 100276, 96, 96, 12288)
# gpt4
f = flops(1024 * 8, 100276, 96, 96, 15447)
f = flops(1024 * 32, 100276, 96, 96, 15447)
f = flops(1024 * 128, 100276, 96, 96, 15447)


