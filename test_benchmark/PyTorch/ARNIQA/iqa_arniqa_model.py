import paddle
import paddle.nn.functional as F
from pd_model.x2paddle_code import Sequential as encoder_paddle_model
from pd_model_regressor.x2paddle_code import TorchLinearRegression as regressor_paddle_model


class ARNIQA(paddle.nn.Layer):

    def __init__(self, default_mean, default_std, feat_dim=2048):
        super(ARNIQA, self).__init__()
        self.default_mean = default_mean
        self.default_std = default_std
        self.feat_dim = feat_dim
        self.encoder = encoder_paddle_model()
        self.regressor = regressor_paddle_model()

        encoder_paddle_params = paddle.load(r'./pd_model/model.pdparams')
        regressor_paddle_params = paddle.load(
            r'./pd_model_regressor/model.pdparams')

        self.encoder.set_dict(encoder_paddle_params, use_structured_name=True)
        self.regressor.set_dict(regressor_paddle_params,
                                use_structured_name=True)

    def forward(self, x) -> float:
        x, x_ds = self._preprocess(x)

        f = F.normalize(self.encoder(x), axis=1)
        f_ds = F.normalize(self.encoder(x_ds), axis=1)
        f_combined = paddle.hstack((f, f_ds)).reshape([-1, self.feat_dim * 2])

        score = self.regressor(f_combined)
        score = self._scale_score(score)

        return score

    def _preprocess(self, x):
        x_ds = F.interpolate(x,
                             scale_factor=0.5,
                             mode="bilinear",
                             align_corners=False)
        x = (x - self.default_mean) / self.default_std
        x_ds = (x_ds - self.default_mean) / self.default_std
        return x, x_ds

    def _scale_score(self, score):
        new_range = (0., 1.)

        # Compute scaling factors
        original_range = (1, 100)
        original_width = original_range[1] - original_range[0]
        new_width = new_range[1] - new_range[0]
        scaling_factor = new_width / original_width

        # Scale score
        scaled_score = new_range[0] + (score -
                                       original_range[0]) * scaling_factor

        return scaled_score
