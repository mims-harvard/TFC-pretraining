import numpy as np
import torch


def DataTransform(sample, config):

    #weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    weak_aug = sample
    #strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    strong_aug = jitter(sample, config.window_len, config.augmentation.jitter_ratio)
    return weak_aug, strong_aug

def jitter(x, window_len, sigma=0.8):
    sub_len = int(0.7 * window_len)
    last_time_idx = window_len - sub_len
    noise_mag = 0.2

    # https://arxiv.org/pdf/1706.00527.pdf
    xtilde = x.clone().detach()
    d1, d2, _ = xtilde.shape
    print(xtilde.shape)
    for i in range(d1): # add noise to a segment of each sample at independently random locations
        for j in range(d2):
            begin = np.random.randint(0, last_time_idx)
            xtilde[i,j,begin:begin+sub_len] += noise_mag * np.random.normal(loc=0., scale=sigma, size=(1, 1, sub_len))
    return xtilde
#     return x + np.random.normal(loc=0., scale=sigma, size=x.shape)ç


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

