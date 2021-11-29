def zero_weights_mantissa(model):
    for name, W in model.named_parameters():
        print(name, W.shape, type(W))
