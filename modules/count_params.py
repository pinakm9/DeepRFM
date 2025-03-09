def RC(D, D_r, *args):
    """
    Counts the number of parameters in a Rerservoir Computer (RC) model.

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.

    Returns:
    int: The number of parameters in the model.
    """
    return D_r*(2*D + D_r)

def RC_s(D, D_r, rho, *args):
    """
    Counts the number of parameters in a sparse  Rerservoir Computer (RC) model.

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    rho (float): Sparsity parameter.

    Returns:
    int: The number of parameters in the model.
    """
    return D_r*(2*D + rho*D_r)

def UnitaryRNN(D, D_r, *args):
    """
    Counts the number of parameters in a Unitary RNN (URNN) model.

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.

    Returns:
    int: The number of parameters in the model.
    """
    return RC(D, D_r)

def LSTM(D, D_r, B, *args):
    """
    Counts the number of parameters in a Long Short-Term Memory (LSTM) model.

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of features.
    B (int): Number of blocks.

    Returns:
    int: The number of parameters in the model.
    """
    return D_r*(4*B*((D+D_r) + 1) + D)

def LocalLSTM(D, D_r, B, G, I, *args):
    """
    Counts the number of parameters in a Local Long Short-Term Memory (LSTM) model.

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of features.
    B (int): Number of blocks.
    G (int): Dimension of the local state.
    I (int): Interaction length.

    Returns:
    int: The number of parameters in the model.
    """
    d = (2*I + 1)*G
    Ng = int(D / G)
    return D_r*(4*B*((d+D_r) + 1) + d)*Ng

def LocalRC(D, D_r, B, G, I, *args):
    """
    Counts the number of parameters in a Local Reservoir Computer (RC) model.

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    B (int): Number of blocks.
    G (int): Dimension of the local state.
    I (int): Interaction length.

    Returns:
    int: The number of parameters in the model.
    """

    d = (2*I + 1)*G
    Ng = int(D / G)
    return D_r*(2*d + D_r)*Ng

def LocalRC_s(D, D_r, B, G, I, rho, *args):
    """
    Counts the number of parameters in a sparse Local Reservoir Computer (RC) model.

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    B (int): Number of blocks.
    G (int): Dimension of the local state.
    I (int): Interaction length.
    rho (float): Sparsity parameter.

    Returns:
    int: The number of parameters in the model.
    """

    d = (2*I + 1)*G
    Ng = int(D / G)
    return D_r*(2*d + rho*D_r)*Ng

def GRU(D, D_r, B):
    """
    Counts the number of parameters in a Gated Recurrent Unit (GRU) model.

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of features.
    B (int): Number of blocks.

    Returns:
    int: The number of parameters in the model.
    """
    return D_r*(3*B*((D+D_r) + 1) + D)

def RFM(D, D_r, *args):
    """
    Counts the number of parameters in a Random Feature Model (RFM).

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.

    Returns:
    int: The number of parameters in the model.
    """
    return D_r*(2*D + 1)

def SkipRFM(D, D_r, *args):
    """
    Counts the number of parameters in a Skip Random Feature Model (SkipRFM).

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.

    Returns:
    int: The number of parameters in the model.
    """
    return RFM(D, D_r)


def DeepRFM(D, D_r, B, *args):
    """
    Counts the number of parameters in a Deep RFM Random Feature Model (DeepSkip).

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    B (int): Number of blocks.

    Returns:
    int: The number of parameters in the model.
    """
    return  D_r*B*(3*D + 1)


def DeepSkip(D, D_r, B, *args):
    """
    Counts the number of parameters in a Deep Skip Random Feature Model (DeepSkip).

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    B (int): Number of blocks.

    Returns:
    int: The number of parameters in the model.
    """
    return  D_r*B*(3*D + 1)

def LocalSkip(D, D_r, B, G, I, *args):
    """
    Counts the number of parameters in a Local Skip Random Feature Model (LocalSkip).

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    B (int): Number of blocks.
    G (int): Dimension of the local state.
    I (int): Interaction length.

    Returns:
    int: The number of parameters in the model.
    """
    d = (2*I + 1)*G
    return D_r*(d + G + 1)

def LocalRFM(D, D_r, B, G, I, *args):
    """
    Counts the number of parameters in a Local Random Feature Model (LocalRFM).

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    B (int): Number of blocks.
    G (int): Dimension of the local state.
    I (int): Interaction length.

    Returns:
    int: The number of parameters in the model.
    """
    d = (2*I + 1)*G
    return D_r*(d + G + 1)

def LocalDeepSkip(D, D_r, B, G, I, *args):
    """
    Counts the number of parameters in a Local Deep Skip Random Feature Model (LocalDeepSkip).

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    B (int): Number of blocks.
    G (int): Dimension of the local state.
    I (int): Interaction length.

    Returns:
    int: The number of parameters in the model.
    """
    d = (2*I + 2)*G
    return D_r*B*(d + G + 1)

def LocalDeepRFM(D, D_r, B, G, I, *args):
    """
    Counts the number of parameters in a Local Deep Random Feature Model (LocalDeepRFM).

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    B (int): Number of blocks.
    G (int): Dimension of the local state.
    I (int): Interaction length.

    Returns:
    int: The number of parameters in the model.
    """
    return LocalDeepSkip(D, D_r, B, G, I, *args)

def LocalDeepRFMN(D, D_r, B, G, I, *args):
    """
    Counts the number of parameters in a Local Deep Random Feature Model trained with artificially noisy data (LocalDeepRFMN).

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    B (int): Number of blocks.
    G (int): Dimension of the local state.
    I (int): Interaction length.

    Returns:
    int: The number of parameters in the model.
    """

    return LocalDeepSkip(D, D_r, B, G, I, *args)

def LocalDeepSkipN(D, D_r, B, G, I, *args):
    """
    Counts the number of parameters in a Local Deep Skip Random Feature Model trained with artificially noisy data (LocalDeepSkipN).

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    B (int): Number of blocks.
    G (int): Dimension of the local state.
    I (int): Interaction length.

    Returns:
    int: The number of parameters in the model.
    """
    return LocalDeepSkip(D, D_r, B, G, I, *args)

def LocalRFMN(D, D_r, B, G, I, *args):
    """
    Counts the number of parameters in a Local Random Feature Model trained with artificially noisy data (LocalRFMN).

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    B (int): Number of blocks.
    G (int): Dimension of the local state.
    I (int): Interaction length.

    Returns:
    int: The number of parameters in the model.
    """
    return LocalSkip(D, D_r, B, G, I, *args)

def LocalSkipN(D, D_r, B, G, I, *args):
    """
    Counts the number of parameters in a Local Skip Random Feature Model trained with artificially noisy data (LocalSkipN).

    Parameters:
    D (int): Input dimensionality.
    D_r (int): Number of random features.
    B (int): Number of blocks.
    G (int): Dimension of the local state.
    I (int): Interaction length.

    Returns:
    int: The number of parameters in the model.
    """

    return LocalSkip(D, D_r, B, G, I, *args)