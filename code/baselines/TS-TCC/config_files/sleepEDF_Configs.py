
class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.final_out_channels = 128
        self.num_classes = 5
        self.dropout = 0.35

        self.kernel_size = 25
        self.stride = 3
        self.features_len = 10 # This only works for the current transfer learning into epilepsy dataset!
        self.window_len = 178

        # training configs
        self.num_epoch = 3


        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-7

        # data parameters
        self.drop_last = True
        self.batch_size = 128

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5
        self.jitter_ratio = 2
        self.max_seg = 12


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 2
