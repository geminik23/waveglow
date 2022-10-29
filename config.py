

import os
import dotenv
from cmath import sqrt

dotenv.load_dotenv()

DATASET_PATH = os.environ.get('DATASET_PATH')
if DATASET_PATH is None: 
    DATASET_PATH = './data'
    os.makedirs(DATASET_PATH, exist_ok=True)

class Config:
    def __init__(self):
        self.dataset_path = DATASET_PATH

        # params from section 3.3 in original paper
        # dataset
        self.lj_folder_name ="LJSpeech-1.1"
        self.lj_train_audio_length = 16000
        self.sample_rate = 22050
        self.num_workers = 8

        # check point 
        self.check_point_folder = "model_cp"
        self.check_point_file = "model_e_{}.pt"
        self.check_point_period = 100

        # params for model
        self.n_mels = 80
        self.hop_length = 256
        self.win_length = 1024
        self.n_fft = 1024
        self.fmin = 80.0
        self.fmax = 8000.0
        self.power = 1.0

        self.n_flows = 12
        self.n_groups = 8
        self.n_early_every = 4
        self.n_early_size = 2

        self.training_std = sqrt(0.5)

        # params for wavenetlike layer
        self.wn_config = {
            'residual_channels': 512,
            'hidden_channels': 512,
            'skip_channels': 256,
            'kernel_size': 3,
            'max_dilation': 8,
        }


        # for training
        self.epochs = 580000
        self.batch_size = 8
        self.lr = 1e-4
        
        self.train_after = None # checkpoint_filepath
        # self.train_after = "model_cp/model_e_400.pt" # checkpoint_filepath





        