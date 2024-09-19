def rc(D, D_r, B):
    return D_r*(2*D + D_r)

def unitary_rnn(D, D_r, B):
    return rc(D, D_r)

def lstm(D, D_r, B):
    return D_r*(4*B*((D+D_r) + 1) + D)

def gru(D, D_r, B):
    return D_r*(3*B*((D+D_r) + 1) + D)

def rfm(D, D_r, B):
    return D_r*(2*D + 1)

def skip_rfm(D, D_r, B):
    return rfm(D, D_r)

def deep_skip(D, D_r, B):
    return  D_r*B*(3*D + 1)

def local_skip_2_2(D, D_r, B):
    d = (2*2 + 1)*2
    return D_r*(d + 2 + 1)

def local_skip_8_1(D, D_r, B):
    d = (2*1 + 1)*8
    return D_r*(d + 8 + 1)

def local_deep_skip_2_2(D, D_r, B):
    d = (2*2 + 2)*2
    return D_r*B*(d + 2 + 1)

def local_deep_skip_8_1(D, D_r, B):
    d = (2*1 + 2)*8
    return D_r*B*(d + 8 + 1)