from nevergrad.optimization.optimizerlib import base, ParametrizedOnePlusOne, ParametrizedMetaModel, ParametrizedCMA, NonObjectOptimizer, Chaining
from ConfigSpace import Configuration

class MetaModelFmin2:
    name = "MetaModelFmin2_Conf"

    def __init__(self, config: Configuration, name: str = None):
        self.config = config
        self.optimizer = self.configure_optimizer()
        if name is not None:
            self.name = name
    
    def configure_optimizer(self):
        CmaFmin2 = NonObjectOptimizer(method="CmaFmin2", random_restart=self.config["random_restart"])
        return ParametrizedMetaModel(
            multivariate_optimizer=CmaFmin2,
            algorithm=self.config["algorithm"],
            frequency_ratio=self.config["frequency_ratio"]
        )  
    
    def get_optimizer(self):
        return self.optimizer

class MetaModelOnePlusOne:
    name = "MetaModelOnePlusOne_Conf"

    def __init__(self, config: Configuration, name: str = None):
        self.config = config
        self.optimizer = self.configure_optimizer()
        if name is not None:
            self.name = name
    
    def configure_optimizer(self):
        if self.config["noise_handling"] == None:
            noise_handling = None
        else:
            noise_handling = (self.config["noise_handling"], self.config["noise_frequency"])
        
        OnePlusOne = ParametrizedOnePlusOne(
            noise_handling = noise_handling,
            mutation=self.config["mutation"], 
            crossover=self.config["crossover"], 
            use_pareto=self.config["use_pareto"],
            sparse=self.config["sparse"], 
            smoother=self.config["smoother"]
        )
    
        return ParametrizedMetaModel(
            multivariate_optimizer=OnePlusOne,
            algorithm=self.config["algorithm"],
            frequency_ratio=self.config["frequency_ratio"]
        )
    
    def get_optimizer(self):
        return self.optimizer
    
class MetaModel:
    name = "MetaModel_Conf"

    def __init__(self, config: Configuration, name: str = None):
        self.config = config
        self.optimizer = self.configure_optimizer()
        if name is not None:
            self.name = name
    
    def configure_optimizer(self):
        CMA = ParametrizedCMA(
            scale=self.config["scale"],
            elitist=self.config["elitist"],
            popsize=self.config["popsize"],
            popsize_factor=self.config["popsize_factor"],
            diagonal=self.config["diagonal"],
            high_speed=self.config["high_speed"],
            fcmaes=self.config["fcmaes"],
            random_init=self.config["random_init"],
        )
        return ParametrizedMetaModel(
            multivariate_optimizer=CMA,
            algorithm=self.config["algorithm"],
            frequency_ratio=self.config["frequency_ratio"]
        )
    
    def get_optimizer(self):
        return self.optimizer
    
class CMA:
    name = "CMA_Conf"

    def __init__(self, config: Configuration, name: str = None):
        self.config = config
        self.optimizer = self.configure_optimizer()
        if name is not None:
            self.name = name
    
    def configure_optimizer(self):
        return ParametrizedCMA(
            scale=self.config["scale"],
            elitist=self.config["elitist"],
            popsize=self.config["popsize"],
            popsize_factor=self.config["popsize_factor"],
            diagonal=self.config["diagonal"],
            high_speed=self.config["high_speed"],
            fcmaes=self.config["fcmaes"],
            random_init=self.config["random_init"],
        )
    
    def get_optimizer(self):
        return self.optimizer
    
class ChainMetaModelPowell:
    name = "ChainMetaModelPowell_Conf"

    def __init__(self, config: Configuration, name: str = None):
        self.config = config
        self.optimizer = self.configure_optimizer()
        if name is not None:
            self.name = name
    
    def configure_optimizer(self):
        CMA = ParametrizedCMA(
            scale=self.config["scale"],
            elitist=self.config["elitist"],
            popsize=self.config["popsize"],
            popsize_factor=self.config["popsize_factor"],
            diagonal=self.config["diagonal"],
            high_speed=self.config["high_speed"],
            fcmaes=self.config["fcmaes"],
            random_init=self.config["random_init"],
        )
        MetaModel = ParametrizedMetaModel(
            multivariate_optimizer=CMA,
            algorithm=self.config["algorithm"],
            frequency_ratio=self.config["frequency_ratio"]
        )
        Powell = NonObjectOptimizer(method="Powell", random_restart=self.config["random_restart"])
        return Chaining([MetaModel, Powell], ["half"])
    
    def get_optimizer(self):
        return self.optimizer

class Cobyla:
    name = "Cobyla"

    def __init__(self):
        self.optimizer = self.configure_optimizer()
    
    def configure_optimizer(self):
        return NonObjectOptimizer(method="COBYLA", random_restart=False)

    def get_optimizer(self):
        return self.optimizer