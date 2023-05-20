import torch

state_dict = torch.load('/shared/bfshi/projects/Robust-Vision-Transformer/output/vit-large/ImageNet21k_pretrained/pytorch_model.bin')

# remove classification head
state_dict.pop('pooler.dense.weight')
state_dict.pop('pooler.dense.bias')

# # remove 'transformer' prefix
# state_dict = {k.replace('vit.', ''): v for k, v in state_dict.items()}

new_state_dict = {}

# pos embed
new_state_dict['pos_embed'] = state_dict['embeddings.position_embeddings']
# cls token
new_state_dict['cls_token'] = state_dict['embeddings.cls_token']
# patch embed
new_state_dict['patch_embed.proj.weight'] = state_dict['embeddings.patch_embeddings.projection.weight']
new_state_dict['patch_embed.proj.bias'] = state_dict['embeddings.patch_embeddings.projection.bias']
# norm
new_state_dict['norm.weight'] = state_dict['layernorm.weight']
new_state_dict['norm.bias'] = state_dict['layernorm.bias']

# each transformer block
for i in range(24):
    new_state_dict[f'blocks.{i}.norm1.weight'] = state_dict[f'encoder.layer.{i}.layernorm_before.weight']
    new_state_dict[f'blocks.{i}.norm1.bias'] = state_dict[f'encoder.layer.{i}.layernorm_before.bias']
    new_state_dict[f'blocks.{i}.norm2.weight'] = state_dict[f'encoder.layer.{i}.layernorm_after.weight']
    new_state_dict[f'blocks.{i}.norm2.bias'] = state_dict[f'encoder.layer.{i}.layernorm_after.bias']
    new_state_dict[f'blocks.{i}.attn.qkv.weight'] = torch.cat([state_dict[f'encoder.layer.{i}.attention.attention.query.weight'],
                                                               state_dict[f'encoder.layer.{i}.attention.attention.key.weight'],
                                                               state_dict[f'encoder.layer.{i}.attention.attention.value.weight']], dim=0)
    new_state_dict[f'blocks.{i}.attn.qkv.bias'] = torch.cat([state_dict[f'encoder.layer.{i}.attention.attention.query.bias'],
                                                               state_dict[f'encoder.layer.{i}.attention.attention.key.bias'],
                                                               state_dict[f'encoder.layer.{i}.attention.attention.value.bias']], dim=0)
    new_state_dict[f'blocks.{i}.attn.proj.weight'] = state_dict[f'encoder.layer.{i}.attention.output.dense.weight']
    new_state_dict[f'blocks.{i}.attn.proj.bias'] = state_dict[f'encoder.layer.{i}.attention.output.dense.bias']
    new_state_dict[f'blocks.{i}.mlp.fc1.weight'] = state_dict[f'encoder.layer.{i}.intermediate.dense.weight']
    new_state_dict[f'blocks.{i}.mlp.fc1.bias'] = state_dict[f'encoder.layer.{i}.intermediate.dense.bias']
    new_state_dict[f'blocks.{i}.mlp.fc2.weight'] = state_dict[f'encoder.layer.{i}.output.dense.weight']
    new_state_dict[f'blocks.{i}.mlp.fc2.bias'] = state_dict[f'encoder.layer.{i}.output.dense.bias']


torch.save(new_state_dict, 'vit-l-in21k.pth')