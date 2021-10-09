"""
Example from 
    ray-project: https://github.com/ray-project/ray


import torch
from ray import tune


def objective(step, alpha, beta):
    return (0.1 + alpha * step / 100)**(-1) + beta * 0.1


def training_function(config):
    # Hyperparameters
    alpha, beta = config["alpha"], config["beta"]
    for step in range(10):
        # Iterative training function - can be any arbitrary training procedure.
        intermediate_score = objective(step, alpha, beta)
        # Feed the score back back to Tune.
        tune.report(mean_loss=intermediate_score)


analysis = tune.run(
    training_function,
    config={
        "alpha": tune.grid_search([0.001, 0.01, 0.1]),
        "beta": tune.choice([1, 2, 3])
    })

print("Best config: ", analysis.get_best_config(metric="mean_loss", mode="min"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
"""

import torch
from .trainer import Trainer
from ray import tune


class AutoTuner:
    def __init__(self) -> None:
        pass

    def run(self, config):
        analysis = tune.run(
        training_function,
        config={
            "alpha": tune.grid_search([0.001, 0.01, 0.1]),
            "beta": tune.choice([1, 2, 3])
        })

class TuneTrainer(Trainer):
    def __init__(self) -> None:
        super().__init__()
    
    def on_after_training(self, **kwargs):
        tune.report(loss=kwargs['loss'])

def training_function(config):    
    model = Model(5, 2)
    dataset = RandomDataset(5, 90)
    trainer = TuneTrainer(config=config)
    trainer.fit(model, dataset)




if __name__ == '__main__':
    
    from torch.utils.data import DataLoader, Dataset    
    # ------------  For debug  --------------
    class RandomDataset(Dataset):
        """ Just For Testing"""
        def __init__(self, size, length):
            self.len = length
            self.data = torch.randn(length, size).to('cuda')

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return self.len


    class Model(nn.Module):
        def __init__(self, input_size, output_size):
            super(Model, self).__init__()
            self.fc = nn.Linear(input_size, output_size)

        def forward(self, input):
            output = self.fc(input)
            print("  In Model: input size", input.size(),
                  "output size", output.size())
            return output

    model = Model(5, 2)
    dataset = RandomDataset(5, 90)
    trainer = Trainer(config={"training":{"batch_size":256, "num_epochs":500}})
    trainer.fit(model, dataset)