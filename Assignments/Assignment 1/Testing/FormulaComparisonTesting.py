import numpy as np
import random

# x = <R_c, G_c, B_c, alpha>

# A = <[I3, <C_k1>],[I3,<C_k2>]>
# Defining some constants

# Template for the matrix we're taking the inverse of
ID3 = np.identity(3)
ID3_Stacked = np.vstack((ID3, ID3))

def pinv_method(Ck1, Ck2, Cf1, Cf2):

    Ck_vector = np.hstack((Ck1, Ck2)).reshape((6, 1))
    Cf_vector = np.hstack((Cf1, Cf2)).reshape((6, 1))
    diff_vector_matrix = np.asmatrix(Cf_vector - Ck_vector)


    A = np.hstack((ID3_Stacked, Ck_vector))
    pinv_A = np.linalg.pinv(A)
    pinv_A_matrix = np.asmatrix(pinv_A)

    return -(pinv_A_matrix * diff_vector_matrix)[3]

def fast_method(Ck1, Ck2, Cf1, Cf2):
    back_diff = (Ck1 - Ck2).astype(np.float32)
    fore_diff = (Cf1 - Cf2).astype(np.float32)

    nume = fore_diff.dot(back_diff)
    denom = back_diff.dot(back_diff)
    alpha = 1 - (nume / denom)

    return alpha

def gen_rand_np_3_arr():
    return np.array([random.randint(0, 255),
                     random.randint(0, 255),
                     random.randint(0, 255)])

if __name__ == '__main__':

    for i in range(10000):
        tup_to_analyze = (gen_rand_np_3_arr(),
                          gen_rand_np_3_arr(),
                          gen_rand_np_3_arr(),
                          gen_rand_np_3_arr())
        pinv_result = pinv_method(*tup_to_analyze)
        fast_result = fast_method(*tup_to_analyze)
        error = abs(pinv_result - fast_result)
        if error > 0.1:
            print pinv_result, fast_result