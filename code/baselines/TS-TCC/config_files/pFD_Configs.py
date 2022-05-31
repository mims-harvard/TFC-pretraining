
class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 4
        self.final_out_channels = 128
        self.features_len = 162

        self.num_classes = 3
        self.dropout = 0.35

        # for noisy labels experiment
        self.corruption_prob = 0.3


        # training configs
        self.num_epoch = 40
        self.batch_size = 64

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 2
        self.jitter_ratio = 0.1
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True

class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 50