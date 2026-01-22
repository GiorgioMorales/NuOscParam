import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from NuOscParam.CalibratorUQ import main_calibrate

if __name__ == '__main__':
    main_calibrate(param="theta_12", mode="flux")
