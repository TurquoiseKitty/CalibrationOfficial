from trainer import grid_searcher


if __name__ == "__main__":

    for dataname in ["boston", "concrete", "energy", "kin8nm","naval", "power", "wine", "yacht"]:
        for modelname in [ "GPmodel", "HNN", "MC_drop", "DeepEnsemble", "HNN_BeyondPinball", "vanillaPred"]:

            print("model: "+ modelname +" on data: "+dataname)
            grid_searcher(
                dataset_name = dataname,
                num_repeat = 5,
                model_name = modelname,
                to_search = {
                    "LR": [1E-2, 5E-3],
                    "bat_size": [10, 64]
                }
            )
