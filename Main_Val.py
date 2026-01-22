import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from NuOscParam.Validator import main_validate

if __name__ == '__main__':
    main_validate(param="theta_12", mode="earth")
