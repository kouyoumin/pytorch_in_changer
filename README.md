# pytorch_in_changer

Finds first conv layer in the model and change in_channels as you need. Useful to convert a RGB model (in_channels=3) to grayscale model (in_channels=1).
## Highlights:
- Reduces computation when you have grayscale input.
- Output is (almost) equivalent to original model (when channel copy applied to grayscale data).