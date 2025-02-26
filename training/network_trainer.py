from re import A

from challenge.test_model_local import load_weights
from scipy.stats import alpha
from networks.model import BlendMLP
from utilities import batch_preprocessing, challenge_metric_loss, sparsity_loss
import numpy as np
import torch
import logging
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class TrainingConfig:
    batch_size = -1
    n_epochs_stop = -1
    num_epochs = -1
    lr_rate = 0.01
    criterion = None
    optimizer = None
    device=None
    def __init__(self, batch_size, n_epochs_stop, num_epochs, lr_rate, criterion, optimizer, device):
        self.batch_size=batch_size
        self.n_epochs_stop=n_epochs_stop
        self.num_epochs=num_epochs
        self.lr_rate=lr_rate
        self.criterion=criterion
        self.optimizer=optimizer
        self.device=device




class NetworkTrainer:
    min_val_loss = 999
    selected_classe = []
    training_config: TrainingConfig = None
    def __init__(self, selected_classes: list, training_config: TrainingConfig, tensorboardWriter: SummaryWriter, domain_weights_file) -> None:
        self.selected_classe=selected_classes
        self.training_config = training_config
        self.tensorboardWriter = tensorboardWriter
        self.domain_weights_file = domain_weights_file
        _, self.domain_weights = load_weights(domain_weights_file)
        self.domain_weights = torch.from_numpy(self.domain_weights)
        logger.debug(f"Initiated NetworkTrainer object\n {self}")


    def train_network(self, model, training_data_loader, epoch, include_domain=True):
        logger.info(f"...{epoch}/{self.training_config.num_epochs}")
        local_step = 0
        epoch_loss = []
        model.to(self.training_config.device)
        for batch in training_data_loader:
            local_step += 1
            model.train()
            alpha_input, beta_input, gamma_input, delta_input, epsilon_input, zeta_input, y = batch_preprocessing(batch, include_domain)
            forecast = model(alpha_input.to(self.training_config.device), beta_input.to(self.training_config.device), gamma_input.to(self.training_config.device), delta_input.to(self.training_config.device), epsilon_input.to(self.training_config.device), zeta_input.to(self.training_config.device))

            loss = self.training_config.criterion(forecast, y.to(self.training_config.device))  # torch.zeros(size=(16,)))
            epoch_loss.append(loss)
            self.training_config.optimizer.zero_grad()
            loss.backward()
            self.training_config.optimizer.step()
            if local_step % 50 == 0:
                logger.info(f"Training loss at step {local_step} = {loss}")

        logger.debug("Finished epoch training")
        result = torch.mean(torch.stack(epoch_loss))
        return result





    def validate_network(self, model, validation_data_loader, epoch, include_domain=True):
        logger.info(f"Entering validation, epoch: {epoch}")
        epoch_loss = []
        model.to(self.training_config.device)
        with torch.no_grad():
            model.eval()
            for batch in validation_data_loader:
                alpha_input, beta_input, gamma_input, delta_input, epsilon_input, zeta_input, y = batch_preprocessing(batch, include_domain)
                forecast = model(alpha_input.to(self.training_config.device), beta_input.to(self.training_config.device), gamma_input.to(self.training_config.device), delta_input.to(self.training_config.device), epsilon_input.to(self.training_config.device), zeta_input.to(self.training_config.device))

                loss = self.training_config.criterion(forecast, y.to(self.training_config.device))
                epoch_loss.append(loss)
        return torch.mean(torch.stack(epoch_loss))

    def log_weights_to_tensorboard(self, last_layer_weights, classes, epoch):
        X = range(last_layer_weights.shape[1])
        weights_sum_per_column = np.sum(last_layer_weights, axis=0)
        weights_sum_per_row = np.sum(last_layer_weights, axis=1)
        self.tensorboardWriter.add_histogram("Weights/per_column", weights_sum_per_column)
        self.tensorboardWriter.add_histogram("Weights/per_row", weights_sum_per_row)

        fig, axs = plt.subplots(3,1)
        heatmap =axs[0].imshow(last_layer_weights, cmap='plasma')
        axs[0].set_xticks(X, labels=X, rotation=45, ha="right", rotation_mode="anchor")
        axs[0].set_yticks(range(len(classes)), range(len(classes)))
        axs[0].set_title(f"Last linear layer weights")

        axs[1].bar(X, weights_sum_per_column)
        axs[1].set_xticks(X, labels=X, rotation=45, ha="right", rotation_mode="anchor")

        axs[2].bar(range(len(classes)), weights_sum_per_row)
        axs[2].set_xticks(range(len(classes)), labels=classes, rotation=45, ha="right", rotation_mode="anchor")
        fig.tight_layout()

        self.tensorboardWriter.add_figure("Last_layer_weights", fig, epoch)
        plt.close(fig)



    def train(self, blendModel, alpha_config, beta_config, training_data_loader, validation_data_loader, fold, leads, include_domain):
        best_model_name="default_model"
        epochs_no_improve=0
        min_val_loss=999999

        for epoch in range(self.training_config.num_epochs):
            epoch_loss = self.train_network(blendModel, training_data_loader, epoch, include_domain=include_domain)
            epoch_validation_loss = self.validate_network(blendModel, validation_data_loader, epoch, include_domain=include_domain)
            self.tensorboardWriter.add_scalar("Loss/training", epoch_loss, epoch)
            self.tensorboardWriter.add_scalar("Loss/validation", epoch_validation_loss, epoch)
            logger.info(f"Training loss for epoch {epoch} = {epoch_loss}")
            logger.info(f"Validation loss for epoch {epoch} = {epoch_validation_loss}")

            last_layer_weights = torch.clone(blendModel.linear.weight.data).cpu().numpy()
            self.log_weights_to_tensorboard(last_layer_weights, blendModel.classes, epoch)

            if epoch_validation_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = epoch_validation_loss
                logger.info(f'Savining {len(leads)}-lead ECG model, epoch: {epoch}...')
                model_name = f"models_repository/{alpha_config.network_name}_{beta_config.network_name}_{leads}_{time.time()}.th"
                logger.debug(f"saving model: {model_name}")
                self.save(model_name,blendModel, self.training_config.optimizer, list(sorted(blendModel.classes)), leads)
                best_model_name=model_name
            else:
                epochs_no_improve += 1
            if epoch > 10 and epochs_no_improve >= self.training_config.n_epochs_stop:
                logger.warn(f'Early stopping!-->epoch: {epoch}; fold: {fold}')
                break
            logger.info(f"not improving since: {epochs_no_improve}")
        return best_model_name




    def test_network():
        return 0




    def save(self, checkpoint_name, model, optimiser, classes, leads):
        torch.save({
            'classes': classes,
            'leads': leads,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict()
            }, checkpoint_name)



