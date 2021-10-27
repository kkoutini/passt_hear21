import torch

from hear21passt import load_model,get_scene_embeddings,get_timestamp_embeddings


if __name__ == '__main__':
    model = load_model().cuda()
    seconds  = 15
    audio = torch.zeros((3,32000*seconds))
    embed,time_stamps = get_timestamp_embeddings(audio, model)
    print(embed.shape)
    print(time_stamps)
    embed = get_scene_embeddings(audio, model)
    print(embed.shape)
