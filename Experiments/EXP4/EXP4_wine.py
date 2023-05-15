
import os
import yaml
from Experiments.EXP1.trainer import trainer, model_callByName, loss_callByName
from data_utils import get_uci_data, common_processor_UCI, seed_all, california_housing_process, normalize, splitter
from Experiments.EXP1.TestPerform import testPerform_muSigma
import torch
import pandas as pd
import numpy as np


if __name__ == "__main__":

    base_seed = 1234
    num_repeat = 5
    big_df = {}

    err_mu_dic = {}
    err_std_dic = {}

    dataname = "wine"

    x, y = get_uci_data("wine", os.getcwd()+"/Dataset/UCI_datasets")

    test_X, test_Y = get_uci_data("wine-red", os.getcwd()+"/Dataset/UCI_datasets")
    
    # te_idx, tr_idx = splitter(1000, len(y)-1000)

    # test_X, test_Y = x[te_idx], y[te_idx]
    # x, y = x[tr_idx], y[tr_idx]

    x_normed, x_normalizer = normalize(x)

    x = x_normed
    test_X = x_normalizer.transform(test_X)
    





    for modelname in [ "HNN", "MC_drop", "DeepEnsemble"]:

        # train base model
        print("model: "+ modelname +" on data: "+dataname)

        with open(os.getcwd()+"/Experiments/EXP1/config_bin/"+modelname+"_on_wine_config.yml", 'r') as file:
            base_configs = yaml.safe_load(file)

        base_misc_info = base_configs["misc_info"]
        base_train_config= base_configs["training_config"]

        base_misc_info["model_config"]["hidden_layers"] = [10, 5]
        base_misc_info["model_config"]["n_input"] = x.shape[1]

        base_train_config["LR"] = 5E-3
        base_train_config["bat_size"] = 64

        crits_dic = {}

        for k in range(num_repeat):

            base_model = model_callByName[base_misc_info["model_init"]](**base_misc_info["model_config"])

            N_train = int(len(x)*0.9)

            

            train_X, test_X = torch.Tensor(x), torch.Tensor(test_X)
            train_Y, test_Y = torch.Tensor(y).to(torch.device("cuda")), torch.Tensor(test_Y).to(torch.device("cuda"))

            
            trainer(
                raw_train_X = train_X,
                raw_train_Y = train_Y,
                model = base_model,
                training_config = base_train_config,
                harvestor = None,          
                misc_info = base_misc_info,
                diff_trainingset = True
            )
            

        
            record = testPerform_muSigma(test_X, test_Y, model_name= modelname, model = base_model)

            if k == 0:
                for key in record.keys():

                    crits_dic[modelname + "_"+key] = []

            for key in record.keys():

                crits_dic[modelname + "_"+key].append(record[key])

        for key in crits_dic.keys():

            err_mu_dic[key] = (max(crits_dic[key]) + min(crits_dic[key]))/2
            err_std_dic[key] = (max(crits_dic[key]) - min(crits_dic[key]))/2



    if len(big_df) == 0:
        big_df["idxes"] = list(err_mu_dic.keys())

    big_df[dataname +"_mu"] = list(err_mu_dic.values())
    big_df[dataname + "_std"] = list(err_std_dic.values())
        
        
    df = pd.DataFrame.from_dict(big_df)  

    df.to_csv(os.getcwd()+"/Experiments/EXP4/record_bin/wine_benchmarks.csv",index=False)



