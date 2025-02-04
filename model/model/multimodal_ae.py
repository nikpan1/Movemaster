import torch.nn as nn

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x


class SkeletonAutoencoder(nn.Module):
    def __init__(self, f_in, layers, dropout, hidden_units, f_embedding, skel, return_embeddings=False):
        super(SkeletonAutoencoder, self).__init__()

        self.skel = skel  # Only using skeleton input
        self.return_embeddings = return_embeddings
        self.layers = layers
        self.f_in = f_in

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Encoder
        if layers == 2:
            self.enc_fc1 = nn.Linear(f_in, hidden_units, bias=True)
            self.enc_fc2 = nn.Linear(hidden_units, f_embedding, bias=True)
        elif layers == 3:
            self.enc_fc1 = nn.Linear(f_in, hidden_units, bias=True)
            self.enc_fc2 = nn.Linear(hidden_units, hidden_units, bias=True)
            self.enc_fc3 = nn.Linear(hidden_units, f_embedding, bias=True)

        # Decoder
        if layers == 2:
            self.dec_fc1 = nn.Linear(f_embedding, hidden_units, bias=True)
            self.dec_fc2 = nn.Linear(hidden_units, f_in, bias=True)
        elif layers == 3:
            self.dec_fc1 = nn.Linear(f_embedding, hidden_units, bias=True)
            self.dec_fc2 = nn.Linear(hidden_units, hidden_units, bias=True)
            self.dec_fc3 = nn.Linear(hidden_units, f_in, bias=True)

    def forward(self, skel):
        """ Forward pass with only skeleton data. """
        self.skel.set_decode_mode(False)

        # Process skeleton input through the sub-model
        skel = self.skel(skel)

        # Flatten the skeleton features
        skel_size = skel.size()
        skel = skel.view(skel.size(0), -1)

        # Encode
        if self.layers == 2:
            x = self.enc_fc1(skel)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.enc_fc2(x)
        elif self.layers == 3:
            x = self.enc_fc1(skel) # here skel - (128, 1080), skel_size - ([128, 24, 3, 15])
            x = self.relu(x)
            x = self.dropout(x)
            x = self.enc_fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.enc_fc3(x)

        # Return embeddings if needed
        if self.return_embeddings:
            return x

        # Decode
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dec_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dec_fc2(x) if self.layers == 2 else self.dec_fc3(x)

        # Reshape back to skeleton format
        skel = x.view(skel_size)

        self.skel.set_decode_mode(True)
        skel = self.skel(skel)

        return skel
