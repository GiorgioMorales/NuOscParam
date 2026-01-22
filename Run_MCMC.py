import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from NuOscParam.MCMC import RunMCMC

if __name__ == '__main__':
    Validator.main_validate(mode="earth")
