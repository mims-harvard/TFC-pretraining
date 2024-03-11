class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 8  # feature count
        self.transformer_nhead = 2
        self.transformer_num_layers = 2
        self.embedding_len = 160  # final embedding len = embedding_len*2

        # training configs
        self.num_epoch = 10
        
        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4 # original lr: 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 16 
        self.TSlength_aligned = 900  # sequence length 15Hz * 60second
        
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