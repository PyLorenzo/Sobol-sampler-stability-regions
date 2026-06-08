from cobaya.likelihood import Likelihood
import torch
import torch.nn as nn
import pickle
import numpy as np

# 1. We need to redefine the architecture to allow PyTorch to load its weights
class StabilityOracle(nn.Module):
    def __init__(self, input_dim):
        super(StabilityOracle, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

# 2. Create the custom class for Cobaya
class NNFastPrior(Likelihood):
    # These two parameters will be passed from the YAML file
    model_path: str
    scaler_path: str

    def initialize(self):
        """Method executed by Cobaya only once at the start of the chain."""
        # Load the scaler
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        # Load the neural network
        self.nn_model = StabilityOracle(input_dim=4)
        self.nn_model.load_state_dict(torch.load(self.model_path))
        self.nn_model.eval() # Set the model in inference mode

    def get_requirements(self):
        """Communicate to Cobaya which parameters we need before evaluating."""
        return ["Log_aT", "c", "sig","M"]

    def logp(self, **params_values):
        """Method called at each single MCMC step."""
        # Extract the current parameters proposed by the MCMC
        xT = params_values["Log_aT"]
        c = params_values["c"]
        sig=params_values["sig"]
        M=params_values["M"]
        # Format and normalize the data as in the training phase
        X = np.array([[xT, c, sig, M]])
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)

        # Query the oracle
        with torch.no_grad():
            output = self.nn_model(X_t)
            
        # If output (logit) > 0, probability > 50% (Stable)
        # NOTE: You can lower this value (e.g. > -1.0) to be more "tolerant" 
        # and let EFTCAMB check uncertain boundary cases.
        if output.item() > 0:
            return 0.0      # Point accepted: we don't modify the probability
        else:
            return -np.inf  # Point rejected: zero probability, the MCMC stops here