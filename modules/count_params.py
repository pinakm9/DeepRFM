def RC(D, D_r, *args):
    return D_r*(2*D + D_r)

def UnitaryRNN(D, D_r, *args):
    return RC(D, D_r)

def LSTM(D, D_r, B, *args):
    return D_r*(4*B*((D+D_r) + 1) + D)

def GRU(D, D_r, B):
    return D_r*(3*B*((D+D_r) + 1) + D)

def RFM(D, D_r, *args):
    return D_r*(2*D + 1)

def SkipRFM(D, D_r, *args):
    return RFM(D, D_r)

def DeepSkip(D, D_r, B, *args):
    return  D_r*B*(3*D + 1)

def LocalSkip(D, D_r, B, G, I, *args):
    d = (2*I + 1)*G
    return D_r*(d + G + 1)

def LocalDeepSkip(D, D_r, B, G, I, *args):
    d = (2*I + 2)*G
    return D_r*B*(d + G + 1)
