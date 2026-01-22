import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from NuOscParam.Trainer.TrainSHTransformer import main_train

if __name__ == '__main__':
    main_train(
        param='theta_12',
        verbose=True,
        mode='flux',
        epochs=501,
        batch_size=1,
        scratch=True
    )
