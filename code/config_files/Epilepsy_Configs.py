class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 32  #128

        self.num_classes = 2
        self.num_classes_target = 3
        self.dropout = 0.35
        self.features_len = 24
        self.features_len_f = 24 # 13 #self.features_len   # the output results in time domain

        # training configs
        self.num_epoch = 40 # 40
        
        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4  # original lr: 3e-4
        self.lr_f = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 32 #64 #  128
        self.target_batch_size = 16 # the size of target dataset (the # of samples used to fine-tune).

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
        self.use_cosine_similarity_f = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 10
