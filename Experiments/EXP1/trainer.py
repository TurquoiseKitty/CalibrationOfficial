from data_utils import seed_all, splitter, common_processor_UCI, get_uci_data
import os
from src.models import vanilla_predNet
from src.losses import mse_loss, rmse_loss
import torch

def trainer(     # describs a training process
        seed,       # key for the experiment to be reproducible
        raw_train_X,
        raw_train_Y,
        model,
        training_config,    # a dictionary that will provide instructions for the model to train
        harvestor,          # will harvest whatever data that might be useful during the experiment
        misc_info,          # a dictionary that contains additional instructions
):
    
    seed_all(seed)

    assert misc_info["input_x_shape"] == raw_train_X.shape
    assert misc_info["input_y_shape"] == raw_train_Y.shape

    if not model:

        model = misc_info["model_init"](**misc_info["model_config"])

    split_percet = misc_info["val_percentage"]
    N_val = int(split_percet*len(raw_train_Y))

    train_idx, val_idx = splitter(len(raw_train_Y)-N_val, N_val, seed = seed)

    train_X, val_X, train_Y, val_Y = raw_train_X[train_idx], raw_train_X[val_idx], raw_train_Y[train_idx], raw_train_Y[val_idx]

    model.train(train_X, train_Y, val_X, val_Y, **training_config, harvestor = harvestor)

    # saver(model, training_config, harvestor, misc_info)

    if misc_info["save_path_and_name"]:
        torch.save(model.state_dict(), misc_info["save_path_and_name"])





if __name__ == "__main__":

    # a short demo of how everything works

    SEED = 1234

    x, y = get_uci_data(data_name= "wine", dir_name= os.getcwd()+"/Dataset/UCI_datasets")

    train_X, test_X, recal_X, train_Y, test_Y, recal_Y = common_processor_UCI(x, y, recal_percent= 0, seed = SEED)

    # carefully set misc_info
    # ---------------------------------------------------------
    misc_info = {}

    misc_info["input_x_shape"] = train_X.shape
    misc_info["input_y_shape"] = train_Y.shape

    misc_info["val_percentage"] = 0.1

    misc_info["model_init"] = vanilla_predNet

    misc_info["model_config"] = {
        "n_input" : len(train_X[0]),
        "hidden_layers" : [10, 10],
        "n_output" : 1,
        "device" : torch.device('cuda')
    }


    misc_info["save_path_and_name"] = os.getcwd() + "/Experiments/EXP1/model_bin/"+"simple_demo.pth"


    # ---------------------------------------------------------


    # carefully set training_config
    # ---------------------------------------------------------
    training_config = {
        "bat_size" : 64,
        "LR" : 1E-2,
        "Decay" : 1E-4,
        "N_Epoch" : 300,
        "validate_times" : 20,
        "verbose" : True,
        "train_loss" : mse_loss,
        "val_loss_criterias" : {
            "mse" : mse_loss,
            "rmse": rmse_loss
        },
        "early_stopping" : True,
        "patience" : 20,
        "monitor_name" : "mse",
        "backdoor" : None
    }

    # ---------------------------------------------------------


    # carefully set harvestor
    # ---------------------------------------------------------

    harvestor = {
        
        "training_losses": [],
        
        "early_stopped": False,

        "early_stopping_epoch": 0,
        "monitor_name" : "mse",
        "monitor_vals" : [],

        "val_mse": [],
        "val_rmse": []

    }


    # ---------------------------------------------------------


    train_Y = torch.Tensor(train_Y).cuda()

    trainer(
        seed = SEED,
        raw_train_X = train_X,
        raw_train_Y = train_Y,
        model = None,
        training_config = training_config,
        harvestor = harvestor,          
        misc_info = misc_info
    )






    
