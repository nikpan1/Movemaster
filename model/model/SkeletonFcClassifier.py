import torch
import torch.nn as nn

class SkeletonFcClassifier(nn.Module):
    def __init__(self, f_in, num_classes, skeleton_ae_model, dropout=0.0, layers=3, hidden_units=100):
        super(SkeletonFcClassifier, self).__init__()
        self.f_in = f_in
        self.skeleton_ae_model = skeleton_ae_model  # Only using the skeleton autoencoder model
        self.layers = layers

        # Fully connected layers
        if layers == 2:
            self.fc1 = nn.Linear(f_in, hidden_units, bias=True)
            self.fc2 = nn.Linear(hidden_units, num_classes, bias=True)
        elif layers == 3:
            self.fc1 = nn.Linear(f_in, hidden_units, bias=True)
            self.fc2 = nn.Linear(hidden_units, hidden_units, bias=True)
            self.fc3 = nn.Linear(hidden_units, num_classes, bias=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, skel):
        """ Forward pass using only skeleton data. """
        x = self.skeleton_ae_model(skel)  # Process through the skeleton autoencoder

        # Fully connected classification layers
        if self.layers == 2:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
        elif self.layers == 3:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)

        return x
