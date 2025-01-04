import torch.nn as nn


class MultimodalFcClassifier(nn.Module):
    def __init__(self, f_in, num_classes, multimodal_ae_model, dropout, layers, hidden_units):
        super(MultimodalFcClassifier, self).__init__()
        self.f_in = f_in
        self.multimodal_ae_model = multimodal_ae_model

        # here I changed it already to 5
        self.layers = 5
        self.fc1 = nn.Linear(f_in, hidden_units, bias=True)
        self.fc2 = nn.Linear(hidden_units, hidden_units, bias=True)
        self.fc3 = nn.Linear(hidden_units, hidden_units, bias=True)
        self.fc4 = nn.Linear(hidden_units, hidden_units, bias=True)
        self.fc5 = nn.Linear(hidden_units, num_classes, bias=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, skel):
        # Pass input through the multimodal autoencoder
        x = self.multimodal_ae_model(skel)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Layer 1
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 4
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 5 (output layer)
        x = self.fc5(x)

        return x
