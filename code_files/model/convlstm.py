# model/convlstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM Cell implementation.

    Args:
        input_dim (int): Number of channels in the input tensor.
        hidden_dim (int): Number of channels in the hidden state.
        kernel_size (tuple): Size of the convolutional kernel.
        bias (bool): Whether or not to add the bias.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Convolution for combined inputs (input + hidden) to gates
        # Output channels = 4 * hidden_dim (for input, forget, cell, output gates)
        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=4 * hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        # Optional: Layer Normalization for stabilizing cell state
        # Applied spatially across channels. Create dummy tensor to get size.
        # Note: LayerNorm might need adjustment based on expected feature map size
        # self.layer_norm = nn.LayerNorm([hidden_dim, height, width]) # Needs H, W

    def forward(self, input_tensor, cur_state):
        """
        Forward pass for a single time step.

        Args:
            input_tensor (torch.Tensor): Input tensor for the current time step.
                                         Shape (B, C_in, H, W).
            cur_state (tuple): Tuple containing the previous hidden state (h_cur)
                               and cell state (c_cur). Shapes (B, C_hid, H, W).

        Returns:
            tuple: Tuple containing the next hidden state (h_next) and cell state (c_next).
        """
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # Concatenate along channel axis

        combined_conv = self.conv_gates(combined)

        # Split the convolutional output into the four gate components
        # cc_i: input gate, cc_f: forget gate, cc_o: output gate, cc_g: cell gate (candidate)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Apply activations
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Calculate next cell state
        c_next = f * c_cur + i * g

        # Optional: Apply Layer Normalization to cell state
        # if hasattr(self, 'layer_norm'):
        #     c_next = self.layer_norm(c_next)

        # Calculate next hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        """
        Initializes the hidden and cell states with zeros.

        Args:
            batch_size (int): The batch size.
            image_size (tuple): The spatial dimensions (H, W) of the input images.
            device (torch.device): The device to create the tensors on.

        Returns:
            tuple: Initial hidden state (h0) and cell state (c0).
        """
        height, width = image_size
        h0 = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c0 = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return (h0, c0)


class ConvLSTMSeq(nn.Module):
    """
    A sequence model using ConvLSTM for segmentation from stacked inputs.

    Processes a sequence of inputs (e.g., temporal pulses) using an initial
    CNN, followed by ConvLSTM layers, and outputs a segmentation map based
    on the final hidden state.

    Args:
        in_channels (int): Number of channels in each input time step (e.g., 1 for grayscale).
        hidden_dims (list[int]): List of hidden dimensions for each ConvLSTM layer.
        kernel_sizes (list[tuple]): List of kernel sizes for each ConvLSTM layer.
        num_classes (int): Number of output segmentation classes.
        initial_cnn_out_channels (int): Number of channels after the initial CNN block.
        batch_first (bool): If True, input tensor shape is (B, T, C, H, W). Default: True.
    """
    def __init__(self, in_channels=1, hidden_dims=[64, 64], kernel_sizes=[(3, 3), (3, 3)],
                 num_classes=1, initial_cnn_out_channels=32, batch_first=True):
        super().__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes] * len(hidden_dims)
        if len(hidden_dims) != len(kernel_sizes):
            raise ValueError("Length of hidden_dims and kernel_sizes must match.")

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = len(hidden_dims)
        self.num_classes = num_classes
        self.batch_first = batch_first

        # --- Initial CNN Feature Extractor ---
        # Applied to each time step independently before ConvLSTM
        # Example: Simple Conv -> BN -> ReLU block
        self.initial_cnn = nn.Sequential(
            nn.Conv2d(in_channels, initial_cnn_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(initial_cnn_out_channels),
            nn.ReLU(inplace=True)
            # Add more layers here if needed
        )
        current_input_dim = initial_cnn_out_channels

        # --- ConvLSTM Layers ---
        cell_list = []
        for i in range(0, self.num_layers):
            cell_list.append(ConvLSTMCell(input_dim=current_input_dim,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_sizes[i],
                                          bias=True))
            current_input_dim = self.hidden_dims[i] # Input for next layer is hidden of current

        self.cell_list = nn.ModuleList(cell_list)

        # --- Output CNN ---
        # Processes the final hidden state of the last ConvLSTM layer
        self.output_conv = nn.Conv2d(in_channels=self.hidden_dims[-1],
                                     out_channels=num_classes,
                                     kernel_size=1, # 1x1 conv to map to classes
                                     padding=0)

    def forward(self, x, hidden_state=None):
        """
        Forward pass through the ConvLSTM sequence model.

        Args:
            x (torch.Tensor): Input tensor. Shape (B, T, C_in, H, W) if batch_first=True,
                              otherwise (T, B, C_in, H, W).
            hidden_state (list, optional): List of initial hidden/cell states for each layer.
                                           If None, initialized to zeros.

        Returns:
            torch.Tensor: Output segmentation logits. Shape (B, C_out, H, W).
        """
        if self.batch_first:
            # Convert to (T, B, C, H, W) for easier iteration
            x = x.permute(1, 0, 2, 3, 4)

        # Get input dimensions
        seq_len, batch_size, _, height, width = x.size()
        device = x.device
        image_size = (height, width)

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, image_size, device)

        # --- Temporal Processing ---
        last_state_list = [] # Store the last state of each layer

        for t in range(seq_len):
            # Input for the current time step
            step_input_raw = x[t, :, :, :, :] # (B, C_in, H, W)

            # Apply initial CNN
            step_input = self.initial_cnn(step_input_raw) # (B, initial_cnn_out, H, W)

            # Pass through ConvLSTM layers
            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                h, c = self.cell_list[layer_idx](input_tensor=step_input, cur_state=[h, c])
                hidden_state[layer_idx] = (h, c) # Update state for next time step
                step_input = h # Output of current layer is input to next layer

            # We only need the hidden state of the *last* layer from the *last* time step
            if t == seq_len - 1:
                 last_layer_h = hidden_state[-1][0] # Get hidden state 'h' of last layer


        # --- Output Generation ---
        # Apply output convolution to the final hidden state of the last layer
        logits = self.output_conv(last_layer_h) # (B, num_classes, H, W)

        return logits

    def _init_hidden(self, batch_size, image_size, device):
        """Initializes hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size, device))
        return init_states


# --- Alias for consistency with train.py ---
ConvLSTM = ConvLSTMSeq