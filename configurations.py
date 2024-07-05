from ConfigSpace import Configuration

def get_config(model: str, dimension: list[int], budget: int, default: bool = False):
    if default:
        with open(f"configurations/default/{model}.txt", 'r') as file:
            config_string = file.read()
        config_dict = eval(config_string)
        return config_dict
    else:
        with open(f"configurations/{budget}/{'_'.join(map(str, dimension))}/{model}_B_{budget}_D_{'_'.join(map(str, dimension))}.txt", 'r') as file:
            config_string = file.read()
        config_dict = eval(config_string)
        return config_dict
        
    # if model == "MetaModelFmin2":
    #     if default:
    #         return {
    #             "frequency_ratio": 0.9,
    #             "algorithm": "quad",  
    #             "random_restart": False,
    #         }
    #     if budget == 200:
    #         if dimension == [2]:
    #             return {
    #                 'algorithm': 'quad',
    #                 'frequency_ratio': 0.41498599153813615,
    #                 'random_restart': True,
    #             }
    #         # if dimension == [3]:
    #         if dimension == [5]:
    #             return {
    #                 "algorithm": "quad",
    #                 "frequency_ratio": 0.4297764876715786,
    #                 "random_restart": False
    #             }
    
    
    # if model == "MetaModelOnePlusOne":
    #     if default:
    #         return {
    #             "frequency_ratio": 0.9,
    #             "algorithm": "quad", 
    #             "noise_handling": None,  
    #             "noise_frequency": None,
    #             "mutation": "gaussian",
    #             "crossover": False,
    #             "use_pareto": False,
    #             "sparse": False,
    #             "smoother": False
    #         }
    #     if budget == 200:
    #         if dimension == [2]:
    #             return {
    #                 'frequency_ratio': 0.4585213487270567,
    #                 'algorithm': 'quad',
    #                 'noise_handling': 'optimistic',
    #                 'noise_frequency': 0.06799287836134571,
    #                 'mutation': 'cauchy',
    #                 'crossover': False,
    #                 'use_pareto': True,
    #                 'sparse': True,
    #                 'smoother': True,
    #             }
    #         # if dimension == [3]:
    #         if dimension == [5]:
    #             return {
    #                 'algorithm': 'quad',
    #                 'crossover': False,
    #                 'frequency_ratio': 0.5831579909707815,
    #                 'mutation': 'cauchy',
    #                 'noise_frequency': 2.4370022872980974e-05,
    #                 'noise_handling': 'optimistic',
    #                 'smoother': False,
    #                 'sparse': False,
    #                 'use_pareto': True,
    #             }
    
    
    # if model == "MetaModel":
    #     if default:
    #         return {
    #             "algorithm": "quad", 
    #             "diagonal": False,
    #             "elitist": False,
    #             "fcmaes": False,
    #             "frequency_ratio": 0.9,
    #             "high_speed": False,
    #             "popsize": None,  
    #             "popsize_factor": 3.0,
    #             "random_init": False,
    #             "scale": 1.0
    #         }
    #     if budget == 200:
    #         if dimension == [2]:
    #             return {
    #                 'algorithm': 'quad',
    #                 'diagonal': False,
    #                 'elitist': False,
    #                 'fcmaes': False,
    #                 'frequency_ratio': 0.5686299219988478,
    #                 'high_speed': False,
    #                 'popsize': 9,
    #                 'popsize_factor': 9.524519031610804,
    #                 'random_init': False,
    #                 'scale': 5.1391179122896045,
    #             }
    #         # if dimension == [3]:
    #         if dimension == [5]:
    #             return {
    #                 "algorithm": "quad",
    #                 "diagonal": True,
    #                 "elitist": True,
    #                 "fcmaes": False,
    #                 "frequency_ratio": 0.7611737930278725,
    #                 "high_speed": False,
    #                 "popsize": 5,
    #                 "popsize_factor": 6.725879711265702,
    #                 "random_init": False,
    #                 "scale": 6.293148508673458
    #             }
    
    
    # if model == "CMA":
    #     if default:
    #         return {
    #             "diagonal": False,
    #             "elitist": False,
    #             "fcmaes": False,
    #             "high_speed": False,
    #             "popsize": None,
    #             "popsize_factor": 3.0,
    #             "random_init": False,
    #             "scale": 1.0
    #         }
    #     if budget == 200:
    #         if dimension == [2]:
    #             return {
    #                 'diagonal': False,
    #                 'elitist': False,
    #                 'fcmaes': True,
    #                 'high_speed': True,
    #                 'popsize': 10,
    #                 'popsize_factor': 9.229026328815932,
    #                 'random_init': True,
    #                 'scale': 3.7180723370094855,
    #             }
    #         # if dimension == [3]:
    #         if dimension == [5]:
    #             return {
    #                 'diagonal': False,
    #                 'elitist': True,
    #                 'fcmaes': False,
    #                 'high_speed': True,
    #                 'popsize': 4,
    #                 'popsize_factor': 3.7328367193360736,
    #                 'random_init': True,
    #                 'scale': 8.935899311166422,
    #             }
    
    
    # if model == "ChainMetaModelPowell":
    #     if default:
    #         return {
    #             "random_restart": False,
    #             "frequency_ratio": 0.9,
    #             "algorithm": "quad", 
    #             "scale": 1.0,
    #             "elitist": False,
    #             "popsize": None, 
    #             "popsize_factor": 3.0,
    #             "diagonal": False,
    #             "high_speed": False,
    #             "fcmaes": False,
    #             "random_init": False
    #         }
    #     if budget == 200:
    #         if dimension == [2]:
    #             return {
    #                 'random_restart': False,
    #                 'frequency_ratio': 0.7822862779883284,
    #                 'algorithm': 'quad',
    #                 'scale': 8.327136771134416,
    #                 'elitist': False,
    #                 'popsize': 10,
    #                 'popsize_factor': 9.847444610081542,
    #                 'high_speed': False,
    #                 'diagonal': True,
    #                 'fcmaes': False,
    #                 'random_init': False,
    #             }
    #         # if dimension == [3]:
    #         if dimension == [5]:
    #             return {
    #                 'algorithm': 'quad',
    #                 'diagonal': True,
    #                 'elitist': True,
    #                 'fcmaes': False,
    #                 'frequency_ratio': 0.8214957858020665,
    #                 'high_speed': True,
    #                 'popsize': 5,
    #                 'popsize_factor': 4.0680513801765015,
    #                 'random_init': True,
    #                 'random_restart': False,
    #                 'scale': 9.271567886984428,
    #             }
    #     if budget == 1000:
    #         if dimension == [2]:
    #             return {
    #                 'algorithm': 'rf',
    #                 'diagonal': False,
    #                 'elitist': True,
    #                 'fcmaes': False,
    #                 'frequency_ratio': 0.7935627444887626,
    #                 'high_speed': True,
    #                 'popsize': 9,
    #                 'popsize_factor': 3.2548265562758822,
    #                 'random_init': False,
    #                 'random_restart': False,
    #                 'scale': 8.025124689899375,
    #             }
    #         if dimension == [5]:
    #             return {
    #                 'algorithm': 'quad',
    #                 'diagonal': False,
    #                 'elitist': False,
    #                 'fcmaes': False,
    #                 'frequency_ratio': 0.9582243549257697,
    #                 'high_speed': True,
    #                 'popsize': 13,
    #                 'popsize_factor': 6.778308886840042,
    #                 'random_init': False,
    #                 'random_restart': False,
    #                 'scale': 9.827837271242995,
    #             }
