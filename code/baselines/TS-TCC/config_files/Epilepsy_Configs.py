class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 25 # 8 for tstcc, 25 for tssd
        self.stride = 3
        self.final_out_channels = 128

        self.num_classes = 2
        self.dropout = 0.35
        self.features_len = 10 # This only works for the current transfer learning into epilepsy dataset!, 24
        self.window_len = 178

        # training configs
        self.num_epoch = 80

        
        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 16

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.001
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 2 #10
