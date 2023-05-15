
import os
import yaml
from Experiments.EXP1.trainer import trainer, model_callByName, loss_callByName
from data_utils import get_uci_data, common_processor_UCI, seed_all, california_housing_process, normalize, splitter
from Experiments.EXP1.TestPerform import testPerform_muSigma, testPerform_kernel
import torch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor




if __name__ == "__main__":

    base_seed = 1234
    num_repeat = 5
    big_df = {}

    err_mu_dic = {}
    err_std_dic = {}

    dataname = "wine"

    x, y = get_uci_data("wine", os.getcwd()+"/Dataset/UCI_datasets")

    test_X, test_Y = get_uci_data("wine-red", os.getcwd()+"/Dataset/UCI_datasets")

    x_normed, x_normalizer = normalize(x)

    x = x_normed
    test_X = x_normalizer.transform(test_X)
    





    for modelname in [ "RFKernel"]:

        # train base model
        print("model: "+ modelname +" on data: "+dataname)

    
        crits_dic = {}

        for k in range(num_repeat):

    

            N_train = int(len(x)*0.7)
            N_recal = int(len(x)*0.3)

            tr_idx, recal_idx = splitter(N_train, N_recal)
            train_X, train_Y = x[tr_idx], y[tr_idx]

            recal_X, recal_Y = x[recal_idx], y[recal_idx]

            train_X, test_X, recal_X= torch.Tensor(train_X), torch.Tensor(test_X), torch.Tensor(recal_X)
            train_Y, test_Y, recal_Y= torch.Tensor(train_Y).to(torch.device("cuda")), torch.Tensor(test_Y).to(torch.device("cuda")), torch.Tensor(recal_Y).to(torch.device("cuda"))

            depth = 10

            rf_model = RandomForestRegressor(max_depth=depth, random_state=0)
            rf_model.fit(train_X.cpu().numpy(), train_Y.cpu().numpy())
            

            record = testPerform_kernel(
                test_X,
                test_Y,
                recal_X,
                recal_Y,
                model_name = "RFKernel",
                model = rf_model,
                val_criterias = [
                    "MACE_Loss", "AGCE_Loss", "CheckScore"
                ],
                wid = 30 
            )
            

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

    df.to_csv(os.getcwd()+"/Experiments/EXP4/record_bin/wine_RFKernel.csv",index=False)



