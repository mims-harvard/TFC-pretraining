class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 4
        self.dropout = 0.35
        self.features_len = 18

        # training configs
        self.num_epoch = 100# 40

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr =  3e-6 # 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

        """New hyperparameters"""
        self.TSlength_aligned = 1500
        self.lr_f = self.lr
        self.target_batch_size = 41# 82 # 41
        self.increased_dim = 1
        self.final_out_channels = 128
        self.num_classes_target = 3
        self.features_len_f = self.features_len
        self.CNNoutput_channel = 190 # 751#


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 6
