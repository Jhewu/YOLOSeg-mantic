import torch

def modify_YOLO(model):
    old_conv_module = model.model.model.model[0]
    print(old_conv_module)

    # 3. Get the inner nn.Conv2d layer and its weights
    old_nn_conv = old_conv_module.conv
    old_conv_weights = old_nn_conv.weight.data

    # --- Setup for the new Conv module ---

    # 5. Determine the new nn.Conv2d layer parameters
    # c2 (output channels) comes from the old nn.Conv2d layer
    out_channels = old_nn_conv.out_channels
    # Other parameters from the old nn.Conv2d layer
    kernel_size = old_nn_conv.kernel_size
    stride = old_nn_conv.stride
    padding = old_nn_conv.padding
    dilation = old_nn_conv.dilation
    groups = old_nn_conv.groups
    # Note: YOLO's Conv sets bias=False, so we skip transferring it.

    # 6. Create the new 4-channel nn.Conv2d layer
    new_nn_conv = torch.nn.Conv2d(
        in_channels=4,  # Change from 3 to 4
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=False  # Must match the original YOLO structure
    )

    # 7. Initialize the new Conv weights tensor
    # Shape: [out_channels, in_channels (4), kernel_height, kernel_width]
    new_conv_weights = new_nn_conv.weight.data

    # 8. Transfer the original 3-channel weights (RGB)
    new_conv_weights[:, 0:3, :, :] = old_conv_weights[:, 0:3, :, :]

    # 9. Calculate and set the average for the 4th channel
    # Calculate the mean across the input channel dimension (dim=1) of the old weights
    avg_weights = old_conv_weights.mean(dim=1, keepdim=True)
    # Copy the averaged 3-channel weights to the 4th channel (index 3)
    new_conv_weights[:, 3:4, :, :] = avg_weights

    # 10. Reconstruct the entire Conv module (and keep the original BN and ACT)
    # We can't easily instantiate the original 'Conv' class without the source file
    # and 'autopad' function, so we modify the existing module's components.

    # Set the new nn.Conv2d layer into the existing Conv module
    old_conv_module.conv = new_nn_conv

    # The BN and ACT layers remain unchanged, which is correct because they operate
    # on the 'out_channels', which hasn't changed (it's still 16 in your example).
    # You must ensure the BN's num_features matches the Conv's out_channels.

    print("u2705 Model's first 'Conv' module successfully modified.")
    print(f"   - New input channels: 4")
    print(f"   - Original output channels: {out_channels}")
