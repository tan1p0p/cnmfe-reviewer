import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from nn.model import Model
import optuna


class NNDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, device):
        self.x, self.y = x, y
        self.device = device
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, i):
        data = (self.x[0][i].to(self.device), self.x[1][i].to(self.device))
        return data, self.y[i].to(self.device)


def train(model, model_prefix, train_loader, test_loader, device, log_path=None):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = F.binary_cross_entropy
    epochs = 50

    def preprocess(y):
        return torch.round(y[0]), y[1]

    precision = ignite.metrics.Precision(preprocess, average=False)
    recall = ignite.metrics.Recall(preprocess, average=False)
    F1 = (precision * recall * 2 / (precision + recall)).mean()

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            'accuracy': ignite.metrics.Accuracy(preprocess),
            'loss': ignite.metrics.Loss(criterion),
            'precision': precision, 'recall': recall, 'f1': F1,
        },
        device=device)
    tester = create_supervised_evaluator(
        model,
        metrics={
            'accuracy': ignite.metrics.Accuracy(preprocess),
            'loss': ignite.metrics.Loss(criterion),
            'precision': precision, 'recall': recall, 'f1': F1,
        },
        device=device)
    writer = SummaryWriter()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        i = (engine.state.iteration - 1) % len(train_loader) + 1
        if i % 50 == 0:
            print(f"\rEpoch[{engine.state.epoch}] Iteration[{i}/{len(train_loader)}] "
                  f"Loss: {engine.state.output:.2f}", end="")
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    def write_metrics(metrics, writer, mode: str, epoch: int):
        """print metrics & write metrics to log"""
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        log_text = f"{mode} Results - Epoch: {epoch}  " + \
                f"loss: {avg_loss:.4f} " + \
                f"accuracy: {avg_accuracy:.4f} " + \
                f"F1(P, R): {metrics['f1']:.4f} " + \
                f"({metrics['precision']:.4f}, {metrics['recall']:.4f})"
        if log_path is None:
            print(log_text)
        else:
            with open(log_path, 'a') as f:
                f.write(log_text + '\n')
        writer.add_scalar(f"{mode}/avg_loss", avg_loss, epoch)
        writer.add_scalar(f"{mode}/avg_accuracy", avg_accuracy, epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        tester.run(train_loader)
        metrics = tester.state.metrics
        write_metrics(metrics, writer, 'training', engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        write_metrics(metrics, writer, 'validation', engine.state.epoch)

    handler = ModelCheckpoint(dirname='./checkpoints', filename_prefix=model_prefix,
                              n_saved=10, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'mymodel': model})

    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    trainer.run(train_loader, max_epochs=epochs)
    return handler.best_score, model
