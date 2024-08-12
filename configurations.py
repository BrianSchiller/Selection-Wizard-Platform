import numpy as np

def get_config(model: str, dimension: list[int], budget: int, default: bool = False):
    if default:
        with open(f"configurations/default/{model}.txt", 'r') as file:
            config_string = file.read()
        config_dict = eval(config_string)
        if model == "MetaModel" or model == "CMA" or model == "ChainMetaModelPowell":
            config_dict["popsize"] = 4 + int(3 * np.log(dimension))
        return config_dict
    else:
        with open(f"configurations/{budget}/{'_'.join(map(str, dimension))}/{model}_B_{budget}_D_{'_'.join(map(str, dimension))}.txt", 'r') as file:
            config_string = file.read()
        config_dict = eval(config_string)
        return config_dict
    