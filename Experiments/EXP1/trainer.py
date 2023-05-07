from data_utils import seed_all, splitter, common_processor_UCI, get_uci_data
import os
from src.models import vanilla_predNet, MC_dropnet, Deep_Ensemble
from src.losses import mse_loss, rmse_loss, mean_std_norm_loss, mean_std_forEnsemble, BeyondPinball_muSigma
import torch
from src.GPmodels import oneLayer_DeepGP
import time
import copy
import numpy as np
import operator


# these are 6 basic models. The rest implements will sort of bases on them
model_callByName = {
    "vanillaPred": vanilla_predNet,
    "GPmodel": oneLayer_DeepGP,
    "HNN": MC_dropnet,
    "MC_drop": MC_dropnet,
    "DeepEnsemble": Deep_Ensemble,
    "HNN_BeyondPinball": MC_dropnet,
}


def empty_harvestor(harvestor):
    for key in harvestor.keys():
        if isinstance(harvestor[key], list):
            harvestor[key] = []

    if "early_stopped" in harvestor.keys():
        harvestor["early_stopping_epoch"] = 0
        harvestor["early_stopped"] = False





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



def grid_searcher(
    dataset_name = "wine",
    dataset_path = os.getcwd()+"/Dataset/UCI_datasets",
    starting_seed = 1234,
    num_repeat = 5,
    model_name = "vanillaPred",
    to_search = {
        "LR": [1E-2, 5E-3, 1E-3],
        "bat_size": [10, 64]
    },
    misc_preconfigs = {
        "val_percentage": 0.1,
        "model_config" :{
            "device" : torch.device('cuda'),
            "hidden_layers" : [10, 5]
        },
        "save_path_and_name": None
    },
    train_preconfig = {
        "Decay" : 1E-4,
        "N_Epoch" : 200,
        "validate_times" : 20,
        "verbose" : False,
        "val_loss_criterias" : {
            "rmse": rmse_loss
        },
        "early_stopping" : True,
        "patience" : 20,
        "backdoor" : None
    },
    harvestor_preconfig = {
        
        "training_losses": [],
        
        "early_stopped": False,

        "early_stopping_epoch": 0,
        "monitor_vals" : [],

    },
    report_path = os.getcwd()+"/Experiments/EXP1/reports/"

):
    
    start_time = time.time()
    
    summarizer = []

    assert model_name in model_callByName.keys()

    for k in range(num_repeat):

        SEED = starting_seed + k

        sub_summarizer = {}

        seed_all(SEED)

        x, y = get_uci_data(data_name= dataset_name, dir_name= dataset_path)

        # now that data is not enough, we just use training set for recalibration
        train_X, test_X, recal_X, train_Y, test_Y, recal_Y = common_processor_UCI(x, y, recal_percent= 0, seed = SEED)

        train_X = torch.Tensor(train_X).cuda()
        train_Y = torch.Tensor(train_Y).cuda()

        # carefully set misc_info, as well as harvestor and training_confiig
        # ---------------------------------------------------------
        misc_info = copy.deepcopy(misc_preconfigs)
        harvestor = copy.deepcopy(harvestor_preconfig)
        training_config = copy.deepcopy(train_preconfig)

        misc_info["input_x_shape"] = train_X.shape
        misc_info["input_y_shape"] = train_Y.shape

        misc_info["model_init"] = model_callByName[model_name]

        # model config is a little bit tricky
        misc_info["model_config"]["n_input"] = len(train_X[0])

        if model_name == "HNN":
            
            misc_info["model_config"]["n_output"] = 2
            misc_info["model_config"]["drop_rate"] = 0.


            training_config["train_loss"] = mean_std_norm_loss
            training_config["val_loss_criterias"] = {
                "nll": mean_std_norm_loss,
                "rmse": rmse_loss
            }
            
            training_config["monitor_name"] = "nll"

            harvestor["monitor_name"] = "nll"
            harvestor["val_nll"] = []
            harvestor["val_rmse"] = []
    
        elif model_name == "MC_drop":

            misc_info["model_config"]["n_output"] = 2
            misc_info["model_config"]["drop_rate"] = 0.2


            training_config["train_loss"] = mean_std_norm_loss
            training_config["val_loss_criterias"] = {
                "nll": mean_std_norm_loss,
                "rmse": rmse_loss
            }
            
            training_config["monitor_name"] = "nll"

            harvestor["monitor_name"] = "nll"
            harvestor["val_nll"] = []
            harvestor["val_rmse"] = []

        elif model_name == "DeepEnsemble": 
            
            misc_info["model_config"]["n_output"] = 2
            misc_info["model_config"]["n_models"] = 5


            training_config["train_loss"] = mean_std_forEnsemble
            training_config["val_loss_criterias"] = {
                "nll": mean_std_forEnsemble,
                "rmse": rmse_loss
            }
            
            training_config["monitor_name"] = "nll"

            harvestor["monitor_name"] = "nll"
            harvestor["val_nll"] = []
            harvestor["val_rmse"] = []

        elif model_name == "GPmodel": 

            training_config["train_loss"] = mean_std_norm_loss
            training_config["val_loss_criterias"] = {
                "nll": mean_std_norm_loss,
                "rmse": rmse_loss
            }
            training_config["num_samples"] = 10
            
            training_config["monitor_name"] = "nll"

            harvestor["monitor_name"] = "nll"
            harvestor["val_nll"] = []
            harvestor["val_rmse"] = []

        elif model_name == "HNN_BeyondPinball": 

            misc_info["model_config"]["n_output"] = 2
            misc_info["model_config"]["drop_rate"] = 0.



            training_config["train_loss"] = BeyondPinball_muSigma
            training_config["val_loss_criterias"] = {
                "beyondPinBall": BeyondPinball_muSigma,
                "rmse": rmse_loss
            }
            
            training_config["monitor_name"] = "beyondPinBall"

            harvestor["monitor_name"] = "beyondPinBall"
            harvestor["val_beyondPinBall"] = []
            harvestor["val_rmse"] = []

        elif model_name == "vanillaPred":
            
            misc_info["model_config"]["n_output"] = 1


            training_config["train_loss"] = mse_loss
            training_config["val_loss_criterias"] = {
                "mse": mse_loss,
                "rmse": rmse_loss
            }
            
            training_config["monitor_name"] = "mse"

            harvestor["monitor_name"] = "mse"
            harvestor["val_mse"] = []
            harvestor["val_rmse"] = []

        # ---------------------------------------------------------


        assert ("LR" in to_search.keys()) and ("bat_size" in to_search.keys())

        for LR in to_search["LR"]:
            for bat_size in to_search["bat_size"]:
                training_config["LR"] = LR
                training_config["bat_size"] = bat_size

                # now start training
                empty_harvestor(harvestor)

                trainer(
                    seed = SEED,
                    raw_train_X = train_X,
                    raw_train_Y = train_Y,
                    model = None,
                    training_config = training_config,
                    harvestor = harvestor,          
                    misc_info = misc_info
                )

                # now the summarizer is grepping informations from the harvestor
                sub_summarizer[(LR, bat_size)] = np.mean(harvestor["monitor_vals"][-3:])


        summarizer.append(min(sub_summarizer.items(), key=operator.itemgetter(1))[0])
        
    choice_para = max(set(summarizer), key = summarizer.count)

    choice_LR, choice_bat_size = choice_para

    finish_time = time.time()



    
    # now we need to write a report

    filename = dataset_name + "_report.txt"

    full_filename = report_path + filename

    if os.path.exists(full_filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    handle = open(full_filename, append_write)
    handle.write("\n\n")
    handle.write("# -------------------------------------------------")
    handle.write("\n\n")
    handle.write("On dataset: "+dataset_name +"\n")
    handle.write("We are training model: "+model_name+"\n")
    handle.write("With training X of size: ({0},{1})\n".format(train_X.shape[0],train_X.shape[1]))
    handle.write("We are grid searching for the best LR and bat_size\n")
    handle.write("After {0:.2f} hours of training\n".format((finish_time - start_time)/3600))
    handle.write("We get a few ideal choices for tuple (LR, bat_size)\n")
    for tup in summarizer:
        handle.write("\t ({0}, {1})\n".format(tup[0], tup[1]))
    handle.write("we finally choose ({0}, {1}) as the best hyperparameters\n".format(choice_LR, choice_bat_size))
    handle.write("All configs are recorded into yaml files in the config directory\n")
    handle.write("\n\n")
    handle.write("# -------------------------------------------------")
    handle.write("\n\n")
    handle.close()








'''

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
        "N_Epoch" : 200,
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

'''

if __name__ == "__main__":

    grid_searcher(
        dataset_name = "wine",
        num_repeat = 2,
        model_name = "vanillaPred",
        to_search = {
            "LR": [1E-2, 5E-3],
            "bat_size": [100, 600]
        }
    )




    