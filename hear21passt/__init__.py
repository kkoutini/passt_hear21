from .api import load_model, get_scene_embeddings, get_timestamp_embeddings

__version__ = "0.0.4"

def embeding_size(hop=50,embeding_size=1000):
    embedings = 20 * 60 * (1000/hop)
    return embedings*embeding_size*4 / (1024*1024*1024) # float32 in GB

