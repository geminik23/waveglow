import torch
from torch.utils.data import DataLoader

from trainer import GeneralTrainer
from model import WaveGlow
from utils import weights_init_xavier_uniform
from dataset import load_ljspeech_dataset


##
# Hyperparameters from config
from config import Config
config = Config()


##
# Datasets
trainset, testset = load_ljspeech_dataset(config)

train_loader = DataLoader(trainset, batch_size=config.batch_size, pin_memory=False, shuffle=True, num_workers=config.num_workers)
test_loader = DataLoader(testset, batch_size=config.batch_size, pin_memory=False)#, num_workers=config.num_workers)


##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

## 
# Initialize the model, optimizer, loss function
model = WaveGlow(config.n_mels, config.n_flows, config.n_groups, config.n_early_every, config.n_early_size, win_length=config.win_length, hop_length=config.hop_length, sigma=config.training_std, wn_config=config.wn_config)
loss_func = lambda t, _: t # model will return the loss directly
optimizer_builder = lambda model: torch.optim.Adam(model.parameters(), lr=config.lr)



##
# Trainer
trainer = GeneralTrainer(model, 
                         optimizer_builder, 
                         loss_func,
                         score_metric={},
                         checkpoint_dir=config.check_point_folder)
trainer.set_tqdm_for_notebook(True)


##
# if train_after is not None then load data and continue the train
reset = True
if config.train_after is not None:
    trainer.load_data( config.train_after)
    reset = False


##
# Train
trainer.train(train_loader, val_loader=test_loader, epochs=config.epochs, device=device, reset=reset, cp_filename=config.check_point_file, cp_period=config.check_point_period)