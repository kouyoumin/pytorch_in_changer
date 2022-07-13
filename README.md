## Obsoleted
Please use [TorchModelModifier](https://github.com/kouyoumin/TorchModelModifier) instead.

# pytorch_in_changer

Finds first conv layer in the model and change in_channels as you need. Useful to convert a RGB model (in_channels=3) to grayscale model (in_channels=1).
## Highlights:
- Reduces computation when you have grayscale input.
- Output is (almost) equivalent to original model (when channel copy applied to grayscale data).

## Limitations:
- Does not work with Inception3 because its transform_input always transform images into 3-channel.
