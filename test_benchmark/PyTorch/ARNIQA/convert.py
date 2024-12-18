import pickle
from collections import OrderedDict

import numpy as np

import torch
import torchvision
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer

import paddle

from iqa_arniqa import forward as paddle_forward
from iqa_arniqa_torch import forward as torch_forward

encoder = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
feat_dim = encoder.fc.in_features

print('-' * 20)
print(encoder)

encoder = nn.Sequential(*list(encoder.children())[:-1])

print('=' * 20)
print(encoder)

print(feat_dim)

encoder_state_dict = torch.load('../dataset/ARNIQA/ARNIQA.pth',
                                weights_only=True)

cleaned_encoder_state_dict = OrderedDict()
for key, value in encoder_state_dict.items():
    # Remove the prefix
    if key.startswith("model."):
        new_key = key[6:]
        cleaned_encoder_state_dict[new_key] = value

encoder.load_state_dict(cleaned_encoder_state_dict)
encoder.eval()

regressor: nn.Module = torch.jit.load(
    '../dataset/ARNIQA/regressor_koniq10k.pth'
)  # Load regressor from torch.hub as JIT model
regressor.eval()

print('-' * 20)
print(regressor.biases)
print(regressor.weights)

# TODO(megemini):
# input_data = np.random.rand(1, 3, 256, 256).astype('float32')

input_model = torch.tensor(input_data)
input_model_paddle = paddle.to_tensor(input_data)

save_dir = "pd_model"
jit_type = "trace"

from x2paddle.convert import pytorch2paddle

pytorch2paddle(encoder,
               save_dir,
               jit_type, [input_model],
               disable_feedback=True)

input_data = np.random.rand(1, 2048).astype('float32')
input_regressor = torch.tensor(input_data)
input_regressor_paddle = paddle.to_tensor(input_data)

save_dir = "pd_model_regressor"

pytorch2paddle(regressor,
               save_dir,
               jit_type, [input_regressor],
               disable_feedback=True)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

default_mean = torch.Tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
default_std = torch.Tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)

default_mean_paddle = paddle.to_tensor(IMAGENET_DEFAULT_MEAN).view([1, 3, 1, 1])
default_std_paddle = paddle.to_tensor(IMAGENET_DEFAULT_STD).view([1, 3, 1, 1])

torch_score = torch_forward(input_model, encoder, regressor, default_mean,
                            default_std, feat_dim)

print('-' * 20)
print(torch_score)

from pd_model.x2paddle_code import Sequential as encoder_paddle_model
from pd_model_regressor.x2paddle_code import TorchLinearRegression as regressor_paddle_model

paddle.disable_static()

encoder_paddle = encoder_paddle_model()
regressor_paddle = regressor_paddle_model()

encoder_paddle_params = paddle.load(r'./pd_model/model.pdparams')
encoder_paddle.set_dict(encoder_paddle_params, use_structured_name=True)
encoder_paddle.eval()

regressor_paddle_params = paddle.load(r'./pd_model_regressor/model.pdparams')
regressor_paddle.set_dict(regressor_paddle_params, use_structured_name=True)
regressor_paddle.eval()

paddle_score = paddle_forward(input_model_paddle, encoder_paddle,
                              regressor_paddle, default_mean_paddle,
                              default_std_paddle, feat_dim)

print('-' * 20)
print(paddle_score)
