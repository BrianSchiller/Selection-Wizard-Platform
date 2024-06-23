def get_config(model: str, dimension: list[int], budget: int, default: bool = False):
    
    if model == "MetaModelFmin2":
        if default:
            return {
                "frequency_ratio": 0.9,
                "algorithm": "quad",  
                "random_restart": False,
            }
        if budget == 100:
            if dimension == [2]:
                return {
                    "algorithm": "quad",
                    "frequency_ratio": 0.46537375891928945,
                    "random_restart": True
                }
            # if dimension == [3]:
            # if dimension == [5]:
    
    
    if model == "MetaModelOnePlusOne":
        if default:
            return {
                "frequency_ratio": 0.9,
                "algorithm": "quad", 
                "noise_handling": None,  
                "noise_frequency": None,
                "mutation": "gaussian",
                "crossover": False,
                "use_pareto": False,
                "sparse": False,
                "smoother": False
            }
        if budget == 100:
            if dimension == [2]:
                return {
                    "algorithm": "quad",
                    "crossover": False,
                    "frequency_ratio": 0.9456714238704398,
                    "mutation": "gaussian",
                    "noise_frequency": 0.06645456927655996,
                    "noise_handling": "optimistic",
                    "smoother": False,
                    "sparse": True,
                    "use_pareto": False
                }
            # if dimension == [3]:
            # if dimension == [5]:
    
    
    if model == "MetaModel":
        if default:
            return {
                "algorithm": "quad",  # Default algorithm
                "diagonal": False,
                "elitist": False,
                "fcmaes": False,
                "frequency_ratio": 0.9,
                "high_speed": False,
                "popsize": None,  
                "popsize_factor": 3.0,
                "random_init": False,
                "scale": 1.0
            }
        if budget == 100:
            if dimension == [2]:
                return {
                    "algorithm": "quad",
                    "diagonal": False,
                    "elitist": False,
                    "fcmaes": False,
                    "frequency_ratio": 0.9456715802768529,
                    "high_speed": True,
                    "popsize": 29,
                    "popsize_factor": 4.135025149988991,
                    "random_init": False,
                    "scale": 5.137380028787924
                }
            # if dimension == [3]:
            # if dimension == [5]:
    
    
    if model == "CMA":
        if default:
            return {
                "diagonal": False,
                "elitist": False,
                "fcmaes": False,
                "high_speed": False,
                "popsize": None,
                "popsize_factor": 3.0,
                "random_init": False,
                "scale": 1.0
            }
        if budget == 100:
            if dimension == [2]:
                return {
                    "diagonal": False,
                    "elitist": True,
                    "fcmaes": False,
                    "high_speed": True,
                    "popsize": 13,
                    "popsize_factor": 4.887926670774734,
                    "random_init": False,
                    "scale": 8.757775980722664
                }
            # if dimension == [3]:
            # if dimension == [5]:
    
    
    if model == "ChainMetaModelPowell":
        if default:
            return {
                "random_restart": False,
                "frequency_ratio": 0.9,
                "algorithm": "quad", 
                "scale": 1.0,
                "elitist": False,
                "popsize": None, 
                "popsize_factor": 3.0,
                "diagonal": False,
                "high_speed": False,
                "fcmaes": False,
                "random_init": False
            }
        if budget == 100:
            if dimension == [2]:
                return {
                    "algorithm": "rf",
                    "diagonal": False,
                    "elitist": False,
                    "fcmaes": True,
                    "frequency_ratio": 0.9333826536729701,
                    "high_speed": False,
                    "popsize": 276,
                    "popsize_factor": 8.148420895508014,
                    "random_init": False,
                    "random_restart": False,
                    "scale": 9.610415793809034
                    }
            # if dimension == [3]:
            # if dimension == [5]:
