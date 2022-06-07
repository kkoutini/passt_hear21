# PaSST package for HEAR 2021 NeurIPS Challenge Holistic Evaluation of Audio Representations


This is an implementation for [Efficient Training of Audio Transformers with Patchout](https://arxiv.org/abs/2110.05069) for HEAR 2021 NeurIPS Challenge
Holistic Evaluation of Audio Representations

# CUDA version 
This is an implementation is tested with CUDA version 11.1, and torch installed:
```shell
pip3 install torch==1.8.1+cu111  torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

# Installation 
Install the latest version of this repo:
```shell
pip install -e 'git+https://github.com/kkoutini/passt_hear21@0.0.16#egg=hear21passt' 
```

The models follow the [common API](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api) of HEAR 21 
:
```shell
hear-validator --model hear21passt.base.pt hear21passt.base
hear-validator --model noweights.txt hear21passt.base2level
hear-validator --model noweights.txt hear21passt.base2levelmel
 ```

There are three modules available `hear21passt.base`,`hear21passt.base2level`, `hear21passt.base2levelmel` :
```python
import torch

from hear21passt.base import load_model, get_scene_embeddings, get_timestamp_embeddings

model = load_model().cuda()
seconds = 15
audio = torch.ones((3, 32000 * seconds))*0.5
embed, time_stamps = get_timestamp_embeddings(audio, model)
print(embed.shape)
embed = get_scene_embeddings(audio, model)
print(embed.shape)
```

# Getting the Logits/Class Labels

You can get the logits (before the sigmoid activation) for the 527 classes of audioset:
```pyton
from hear21passt.base import load_model

model = load_model(mode="logits").cuda()
logits = model(wave_signal)
```
The class labels indices can be found [here](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/metadata/class_labels_indices.csv)
