from utilities import batch_preprocessing
import numpy as np
import torch
import logging
import os
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class TrainingConfig:
    batch_size = -1
    n_epochs_stop = -1
    num_epochs = -1
    lr_rate = 0.01
    criterion = None
    optimizer = None
    device=None
    model_repository: str = ""
    def __init__(self, batch_size, n_epochs_stop, num_epochs, lr_rate, criterion, optimizer, device, model_repository):
        self.batch_size=batch_size
        self.n_epochs_stop=n_epochs_stop
        self.num_epochs=num_epochs
        self.lr_rate=lr_rate
        self.criterion=criterion
        self.optimizer=optimizer
        self.device=device
        self.model_repository=model_repository




class NetworkTrainer:
    min_val_loss = 999
    selected_classe = []
    training_config: TrainingConfig = None
    def __init__(self, selected_classes: list, training_config: TrainingConfig, tensorboardWriter: SummaryWriter) -> None:
        self.selected_classe=selected_classes
        self.training_config = training_config
        self.tensorboardWriter = tensorboardWriter
        logger.debug(f"Initiated NetworkTrainer object\n {self}")


    def train_network(self, model, training_data_loader, epoch):
        logger.info(f"...{epoch}/{self.training_config.num_epochs}")
        local_step = 0
        epoch_loss = []
        model.to(self.training_config.device)
        for batch in training_data_loader:
            local_step += 1
            model.train()
            alpha_input, beta_input, gamma_input, delta_input, epsilon_input, y = batch_preprocessing(batch)
            forecast = model(alpha_input.to(self.training_config.device), beta_input.to(self.training_config.device), gamma_input.to(self.training_config.device), delta_input.to(self.training_config.device), epsilon_input.to(self.training_config.device))

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





    def validate_network(self, model, validation_data_loader, epoch):
        logger.info(f"Entering validation, epoch: {epoch}")
        epoch_loss = []
        model.to(self.training_config.device)
        with torch.no_grad():
            model.eval()
            for batch in validation_data_loader:
                alpha_input, beta_input, gamma_input, delta_input, epsilon_input, y = batch_preprocessing(batch)
                forecast = model(alpha_input.to(self.training_config.device), beta_input.to(self.training_config.device), gamma_input.to(self.training_config.device), delta_input.to(self.training_config.device), epsilon_input.to(self.training_config.device))

                loss = self.training_config.criterion(forecast, y.to(self.training_config.device))
                epoch_loss.append(loss)
        return torch.mean(torch.stack(epoch_loss))


    def train(self, blendModel, alpha_config, beta_config, training_data_loader, validation_data_loader, leads):
        model_name = os.path.join(self.training_config.model_repository, "best_model_physionet2025.th")
        epochs_no_improve=0
        min_val_loss=999999

        for epoch in range(self.training_config.num_epochs):
            epoch_loss = self.train_network(blendModel, training_data_loader, epoch)
            epoch_validation_loss = self.validate_network(blendModel, validation_data_loader, epoch)
            self.tensorboardWriter.add_scalar("Loss/training", epoch_loss, epoch)
            self.tensorboardWriter.add_scalar("Loss/validation", epoch_validation_loss, epoch)
            logger.info(f"Training loss for epoch {epoch} = {epoch_loss}")
            logger.info(f"Validation loss for epoch {epoch} = {epoch_validation_loss}")

            if epoch_validation_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = epoch_validation_loss
                logger.info(f'Savining {len(leads)}-lead ECG model, epoch: {epoch}...')
                logger.debug(f"saving model: {model_name}")
                os.makedirs(self.training_config.model_repository, exist_ok=True)
                self.save(model_name,blendModel, self.training_config.optimizer, list(sorted(blendModel.classes)), leads)
            else:
                epochs_no_improve += 1
            if epoch > 10 and epochs_no_improve >= self.training_config.n_epochs_stop:
                logger.warn(f'Early stopping!-->epoch: {epoch}.')
                break
            logger.info(f"not improving since: {epochs_no_improve}")
        return model_name




    def test_network():
        return 0




    def save(self, checkpoint_name, model, optimiser, classes, leads):
        torch.save({
            'classes': classes,
            'leads': leads,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict()
            }, checkpoint_name)



