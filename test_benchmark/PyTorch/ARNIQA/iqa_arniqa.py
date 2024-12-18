import paddle
import paddle.nn.functional as F


def forward(x,
            encoder,
            regressor,
            default_mean,
            default_std,
            feat_dim=2048) -> float:
    x, x_ds = _preprocess(x, default_mean, default_std)

    f = F.normalize(encoder(x), axis=1)
    f_ds = F.normalize(encoder(x_ds), axis=1)
    f_combined = paddle.hstack((f, f_ds)).view([-1, feat_dim * 2])

    score = regressor(f_combined)
    score = _scale_score(score)

    return score


def _preprocess(x, default_mean, default_std):
    x_ds = F.interpolate(x,
                         scale_factor=0.5,
                         mode="bilinear",
                         align_corners=False)
    x = (x - default_mean.to(x)) / default_std.to(x)
    x_ds = (x_ds - default_mean.to(x_ds)) / default_std.to(x_ds)
    return x, x_ds


def _scale_score(score):
    new_range = (0., 1.)

    # Compute scaling factors
    original_range = (1, 100)
    original_width = original_range[1] - original_range[0]
    new_width = new_range[1] - new_range[0]
    scaling_factor = new_width / original_width

    # Scale score
    scaled_score = new_range[0] + (score - original_range[0]) * scaling_factor

    return scaled_score
