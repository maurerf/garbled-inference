# performance benchmarking utlity for Microsoft SEAL. Simulating approximations of nonlinear functions used in CNN
# activation layers.
import datetime
import time
import seal
import numpy as np


def relu_approx_1(evaluator, x):
    """
    Square approximation
    """
    return evaluator.square_inplace(x)


def relu_approx_2(evaluator, x):
    """
    Deg. 2 approximation:
    0.375373 + 0.5x + 0.117071x²
    source: https://arxiv.org/pdf/2009.03727.pdf, table 2
    """
    return evaluator.square_inplace(x)  # todo


def relu_approx_3(evaluator, x):
    """
    Deg. 4 approximation:
    0.234606 + 0.5x + 0.204875x²− 0.0063896x⁴
    source: https://arxiv.org/pdf/2009.03727.pdf, table 2
    """
    return evaluator.square_inplace(x)  # todo


def sign_approx_1(evaluator, x):
    return x  # todo: find approximations for sign


def sign_approx_2(evaluator, x):
    return x  # todo: find approximations for sign


def sign_approx_3(evaluator, x):
    return x  # todo: find approximations for sign


def process(evaluator, encryptor, decryptor, relin_keys, input):
    """
    Processes one single int64 input, i.e. integer equivalent to double input in CNN.
    Applies approximation of activation function on encrypted data point.

    :param evaluator SEAL evaluator
    :param encryptor SEAL encryptor
    :param decryptor SEAL decryptor
    :param input 64 bit integer in NumPy plaintext

    :return SEAL ciphertext of result of approximation
    """
    if evaluator is not None:
        # encode and encrypt
        x_plain = batch_encoder.encode([input])
        x_encrypted = encryptor.encrypt(x_plain)
        # print(f'noise budget in freshly encrypted x: {decryptor.invariant_noise_budget(x_encrypted)}')

        # apply approximation. choose which approximation to use here
        x_res = relu_approx_1(evaluator, x_encrypted)
        # x_res = relu_approx_2(evaluator, x_encrypted)
        # x_res = relu_approx_3(evaluator, x_encrypted)
        # x_res = sign_approx_1(evaluator, x_encrypted)
        # x_res = sign_approx_1(evaluator, x_encrypted)
        # x_res = sign_approx_1(evaluator, x_encrypted)

        # decrypt and decode result
        evaluator.relinearize_inplace(x_res, relin_keys)
        # print(f'noise budget in x_squared: {decryptor.invariant_noise_budget(x_encrypted)} bits')
        return decryptor.decrypt(x_res)
        # print(batch_encoder.decode(decrypted_result))


# entry point for SEAL activation function approximation benchmark
if __name__ == '__main__':
    # start measuring computation time
    start_main = time.time()

    print('SEAL approx. activation function benchmark: start main function')

    # init SEAL param object
    params = seal.EncryptionParameters(seal.scheme_type.bfv)

    # set parameters in param object
    poly_mod_degree = 16384  # = N; source: https://arxiv.org/pdf/1811.00778.pdf, table 6
    plaintext_mod = 5522259017729  # = q; source: https://arxiv.org/pdf/1811.00778.pdf, table 6
    params.set_plain_modulus(plaintext_mod)
    params.set_poly_modulus_degree(poly_mod_degree)
    params.set_coeff_modulus(seal.CoeffModulus.BFVDefault(poly_mod_degree))

    # init SEAL context
    context = seal.SEALContext(params)
    batch_encoder = seal.BatchEncoder(context)

    # init keys
    keygen = seal.KeyGenerator(context)
    secret_key = keygen.secret_key()
    public_key = keygen.create_public_key()
    relin_keys = keygen.create_relin_keys()

    # init evaluator, encryptor and decryptor
    evaluator = seal.Evaluator(context)
    encryptor = seal.Encryptor(context, public_key)
    decryptor = seal.Decryptor(context, secret_key)

    # init randomized input. Inputs shall be larger than 32 bit, i.e. of type np.int64
    MAX_INT32 = 2147483647
    MAX_INT64 = 9223372036854775807
    x_rand = np.random.randint(MAX_INT32, MAX_INT64, 1000, dtype=np.int64)
    print(x_rand)

    # execute and time approximation function
    for x in x_rand:
        process(evaluator, encryptor, decryptor, relin_keys, x)

    end_main = time.time()
    print('SEAL approx. activation function benchmark: end of main function')
    print('time elapsed: ', str(datetime.timedelta(seconds=(end_main - start_main))))
