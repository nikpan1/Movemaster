import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import RandomSampler, ConcatDataset

from model.ConvAutoencoder import ConvAutoencoder
from model.SkeletonAutoencoder import SkeletonAutoencoder
from model.SkeletonFcClassifier import SkeletonFcClassifier
from my_utils.dataset import MMFit, SequentialStridedSampler

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class args:
    """
    Configuration parameters for training and evaluating an activity recognition network. Attributes:
        num_classes : Number of output classes for classification
        window_length : Duration (in seconds) of the temporal data window
        window_stride : Stride (in seconds) between successive data windows (later converted to frame stride)
        skeleton_sampling_rate : Sampling frequency for skeleton data
        epochs : Total number of training epochs
        ae_layers : Number of fully connected layers in the autoencoder
        ae_hidden_units : Number of hidden units per autoencoder layer
        embedding_unit : Number of units in the embedding layer
        ae_dropout : Dropout probability used in the autoencoder
        layers : Number of fully connected layers in the main network
        hidden_units : Number of hidden units per layer in the main network
        dropout : Dropout probability applied in the main network
        lr : Learning rate for optimization
        batch_size : Batch size for training iterations
        eval_every : Frequency (in epochs) at which model evaluation is performed
        early_stop : Number of epochs with no validation improvement before stopping early
        checkpoint : Frequency (in epochs) for saving model checkpoints
        name : name for the experiment
        output : Directory path for saving experiment outputs
        data : File path to the dataset
        device : Computation device
        torch.backends.cudnn.benchmark : enabling cuDNN auto-tuning
        ACTIONS : List of activity labels
        TRAIN_W_IDs : training data splits
        VAL_W_IDs : validation data splits
        TEST_W_IDs : test data splits
        window_stride : Computed stride in frames
        skeleton_window_length : window length in frames
        data_transforms : List of applied transformations to data
    """
    num_classes = 11
    window_length = 3
    window_stride = 0.2
    skeleton_sampling_rate = 30
    epochs = 50
    ae_layers = 3
    ae_hidden_units = 128
    ae_dropout = 0.40
    embedding_units = 86
    layers = 3
    hidden_units = 16
    dropout = 0.40
    lr = 0.0005
    weight_decay = 0.0005
    batch_size = 128
    eval_every = 1
    early_stop = 10
    checkpoint = 1

    skeleton_ae_f_in = 2160
    MODALITY = 'pose_2d'
    ACTIONS = [
        'squats', 'lunges', 'bicep_curls', 'situps', 'pushups', 'tricep_extensions',
        'dumbbell_rows', 'jumping_jacks', 'dumbbell_shoulder_press', 'lateral_shoulder_raises', 'non_activity'
    ]

    TRAIN_W_IDs = ['01', '02', '03', '04', '06', '07', '08', '16', '17', '18']
    VAL_W_IDs = ['14', '15', '19']
    TEST_W_IDs = ['09', '10', '11']

    window_stride = int(window_stride * skeleton_sampling_rate)
    skeleton_window_length = int(window_length * skeleton_sampling_rate)

    # Apply more aggressive augmentations
    from my_utils.data_transforms import Unit, HorizontalFlip
    train_data_transforms = [Unit(), HorizontalFlip()]
    test_data_transforms = [Unit()]

    name = "mmfit_optimized"
    output = "output-optimized/"
    data = "C:/Users/nikod/Desktop/movemaster-nn/mm-fit"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True


class skel_args:
    skeleton_window_length = args.skeleton_window_length
    skeleton_window_height = 31
    input_size = (skeleton_window_length, skeleton_window_height)
    input_channel_size = 2
    dimensions = 2
    layers = 3
    grouped = [3, 3, 1]
    kernel_size = 11
    kernel_stride = (2, 1)


def instantiate_model():
    skel_model = ConvAutoencoder(
        input_size=skel_args.input_size,
        input_ch=skel_args.input_channel_size,
        dim=skel_args.dimensions,
        layers=skel_args.layers,
        grouped=skel_args.grouped,
        kernel_size=skel_args.kernel_size,
        kernel_stride=skel_args.kernel_stride,
        return_embeddings=True,
        decode_only=False
    ).to(args.device, non_blocking=True)

    skeleton_ae_model = SkeletonAutoencoder(
        f_in=args.skeleton_ae_f_in,
        layers=args.layers,
        dropout=args.ae_dropout,
        hidden_units=args.ae_hidden_units,
        f_embedding=args.embedding_units,
        skel=skel_model,
        return_embeddings=True
    ).to(args.device, non_blocking=True)

    model = SkeletonFcClassifier(
        f_in=args.embedding_units,
        num_classes=args.num_classes,
        skeleton_ae_model=skeleton_ae_model,
        layers=args.layers,
        hidden_units=args.hidden_units,
        dropout=args.dropout
    ).to(args.device, non_blocking=True)

    return model


def prepare_train_loaders():
    train_datasets, val_datasets, test_datasets = [], [], []

    # Prepare training datasets with additional augmentation
    for w_id in args.TRAIN_W_IDs:
        modality_filepaths = {}
        workout_path = os.path.join(args.data, 'w' + w_id)
        files = os.listdir(workout_path)
        label_path = None
        for file in files:
            if 'labels' in file:
                label_path = os.path.join(workout_path, file)
                continue
            if args.MODALITY in file:
                modality_filepaths[args.MODALITY] = os.path.join(workout_path, file)

        for transform_instance in args.train_data_transforms:
            dt = transforms.Compose([transform_instance])
            train_datasets.append(
                MMFit(modality_filepaths, label_path, args.window_length, args.skeleton_window_length,
                      skeleton_transform=dt)
            )

    # Prepare validation datasets without augmentation
    for w_id in args.VAL_W_IDs:
        modality_filepaths = {}
        workout_path = os.path.join(args.data, 'w' + w_id)
        files = os.listdir(workout_path)
        label_path = None
        for file in files:
            if 'labels' in file:
                label_path = os.path.join(workout_path, file)
                continue
            if args.MODALITY in file:
                modality_filepaths[args.MODALITY] = os.path.join(workout_path, file)

        for transform_instance in args.test_data_transforms:
            dt = transforms.Compose([transform_instance])
            val_datasets.append(
                MMFit(modality_filepaths, label_path, args.window_length, args.skeleton_window_length,
                      skeleton_transform=dt)
            )

    # Prepare test datasets without augmentation
    for w_id in args.TEST_W_IDs:
        modality_filepaths = {}
        workout_path = os.path.join(args.data, 'w' + w_id)
        files = os.listdir(workout_path)
        label_path = None
        for file in files:
            if 'labels' in file:
                label_path = os.path.join(workout_path, file)
                continue
            if args.MODALITY in file:
                modality_filepaths[args.MODALITY] = os.path.join(workout_path, file)

        for transform_instance in args.test_data_transforms:
            dt = transforms.Compose([transform_instance])
            test_datasets.append(
                MMFit(modality_filepaths, label_path, args.window_length, args.skeleton_window_length,
                      skeleton_transform=dt)
            )

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=True,
        num_workers=8
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialStridedSampler(val_dataset, args.window_stride),
        pin_memory=True,
        num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        sampler=SequentialStridedSampler(test_dataset, args.window_stride),
        pin_memory=True,
        num_workers=8
    )

    return train_loader, val_loader, test_loader


# Mixed Precision Training
scaler = GradScaler()


def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total = 0, 0, 0

    with tqdm.tqdm(total=len(train_loader), desc="Training") as pbar:
        for modalities, labels, reps in train_loader:
            for modality, data in modalities.items():
                modalities[modality] = data.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():  # Mixed precision
                outputs = model(modalities['pose_2d'])
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total += labels.size(0)
            total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

            pbar.update(1)
            pbar.set_postfix({'Acc': f"{total_correct / total:.4f}", 'Loss': f"{total_loss / total:.4f}"})

    return total_loss / total, total_correct / total


def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0

    with torch.no_grad():
        with tqdm.tqdm(total=len(val_loader), desc="Validation") as pbar:
            for modalities, labels, reps in val_loader:
                for modality, data in modalities.items():
                    modalities[modality] = data.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)

                with autocast():
                    outputs = model(modalities['pose_2d'])
                    loss = criterion(outputs, labels)

                total_loss += loss.item()
                total += labels.size(0)
                total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

                pbar.update(1)
                pbar.set_postfix({'Acc': f"{total_correct / total:.4f}", 'Loss': f"{total_loss / total:.4f}"})

    return total_loss / total, total_correct / total


# Training loop
if __name__ == '__main__':
    os.makedirs(args.output, exist_ok=True)

    train_loader, val_loader, test_loader = prepare_train_loaders()
    model = instantiate_model()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_valid_acc = 0.0
    steps_since_improvement = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            best_model_state_dict = copy.deepcopy(model.state_dict())
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        if steps_since_improvement >= args.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        scheduler.step(val_acc)

    torch.save({'model_state_dict': best_model_state_dict}, os.path.join(args.output, "best_model.pth"))
