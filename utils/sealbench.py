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
    evaluator.square_inplace(x)
    return x


def relu_approx_2(evaluator, x):
    """
    Deg. 2 approximation (more precise)
    source: Chou et al., 2018
    """
    # 0.12050344x²
    term1 = seal.Ciphertext()
    term1 = evaluator.multiply(x, x)
    evaluator.multiply_plain_inplace(term1, batch_encoder.encode([quantise(0.12050344, 64)]))

    # 0.5x
    term2 = seal.Ciphertext()
    term2 = evaluator.multiply_plain(x, batch_encoder.encode([quantise(0.5, 64)]))

    # 0.153613744
    term3 = batch_encoder.encode([quantise(0.153613744, 64)])

    # add terms
    res = term1
    evaluator.add_inplace(res, term2)
    evaluator.add_plain_inplace(res, term3)

    return res


def relu_approx_3(evaluator, x):
    """
    Deg. 2 approximation (powers of 2)
    source: Chou et al., 2018
    """
    # 0.125x²
    term1 = seal.Ciphertext()
    term1 = evaluator.multiply(x, x)
    evaluator.multiply_plain_inplace(term1, batch_encoder.encode([quantise(0.125, 64)]))

    # 0.5x
    term2 = seal.Ciphertext()
    term2 = evaluator.multiply_plain(x, batch_encoder.encode([quantise(0.5, 64)]))

    # 1.0
    term3 = batch_encoder.encode([quantise(1.0, 64)])

    # add terms
    res = term1
    evaluator.add_inplace(res, term2)
    evaluator.add_plain_inplace(res, term3)

    return res


def tanh_approx_1(evaluator, x):
    """
    Deg. 2 approximation
    source: Gottemukkula, 2019
    """
    # -0.0000245768494133x²
    term1 = seal.Ciphertext()
    term1 = evaluator.multiply(x, x)
    evaluator.multiply_plain_inplace(term1, batch_encoder.encode([quantise(-0.0000245768494133, 64)]))

    # 0.29x
    term2 = seal.Ciphertext()
    term2 = evaluator.multiply_plain(x, batch_encoder.encode([quantise(0.29, 64)]))

    # 0.0000153605308833
    term3 = batch_encoder.encode([quantise(0.0000153605308833, 64)])

    # add terms
    res = term1
    evaluator.add_inplace(res, term2)
    evaluator.add_plain_inplace(res, term3)

    return res


def tanh_approx_2(evaluator, x):
    """
    Deg. 3 approximation
    source: Gottemukkula, 2019
    """
    # -0.01x³
    term1 = seal.Ciphertext()
    term1 = evaluator.multiply(x, x)
    term1 = evaluator.multiply(term1, x)
    evaluator.multiply_plain_inplace(term1, batch_encoder.encode([quantise(-0.01, 64)]))

    # -0.0000998798454775x²
    term2 = seal.Ciphertext()
    term2 = evaluator.multiply(x, x)
    evaluator.multiply_plain_inplace(term2, batch_encoder.encode([quantise(-0.0000998798454775, 64)]))

    # 0.51x
    term3 = seal.Ciphertext()
    term3 = evaluator.multiply_plain(x, batch_encoder.encode([quantise(0.51, 64)]))

    # 0.0001234098040867
    term4 = batch_encoder.encode([quantise(0.0001234098040867, 64)])

    # add terms
    res = term1
    evaluator.add_inplace(res, term2)
    evaluator.add_inplace(res, term3)
    evaluator.add_plain_inplace(res, term4)

    return res


def tanh_approx_3(evaluator, x):
    """
    Deg. 4 approximation
    source: Gottemukkula, 2019
    """
    # -0.0000680998946437x⁴
    term1 = seal.Ciphertext()
    term1 = evaluator.multiply(x, x)
    term1 = evaluator.multiply(term1, term1)
    evaluator.multiply_plain_inplace(term1, batch_encoder.encode([quantise(-0.0000680998946437, 64)]))

    # -0.01x³
    term2 = seal.Ciphertext()
    term2 = evaluator.multiply(x, x)
    term2 = evaluator.multiply(term2, x)
    evaluator.multiply_plain_inplace(term2, batch_encoder.encode([quantise(-0.01, 64)]))

    # -0.0005553441183901x²
    term3 = seal.Ciphertext()
    term3 = evaluator.multiply(x, x)
    evaluator.multiply_plain_inplace(term3, batch_encoder.encode([quantise(-0.0005553441183901, 64)]))

    # 0.51x
    term4 = seal.Ciphertext()
    term4 = evaluator.multiply_plain(x, batch_encoder.encode([quantise(0.51, 64)]))

    # 0.0001234098040867
    term5 = batch_encoder.encode([quantise(0.0003690088906928, 64)])

    # add terms
    res = term1
    evaluator.add_inplace(res, term2)
    evaluator.add_inplace(res, term3)
    evaluator.add_inplace(res, term4)
    evaluator.add_plain_inplace(res, term5)

    return res


def swish_approx_1(evaluator, x):
    """
    Deg. 2 approximation
    source: Gottemukkula, 2019
    """
    # 0.1²
    term1 = seal.Ciphertext()
    term1 = evaluator.multiply(x, x)
    evaluator.multiply_plain_inplace(term1, batch_encoder.encode([quantise(0.1, 64)]))

    # 0.5x
    term2 = seal.Ciphertext()
    term2 = evaluator.multiply_plain(x, batch_encoder.encode([quantise(0.5, 64)]))

    # 0.24
    term3 = batch_encoder.encode([quantise(0.24, 64)])

    # add terms
    res = term1
    evaluator.add_inplace(res, term2)
    evaluator.add_plain_inplace(res, term3)

    return res


def swish_approx_2(evaluator, x):
    """
    Deg. 3 approximation
    source: Gottemukkula, 2019
    """
    # -0.000054479915715x³
    term1 = seal.Ciphertext()
    term1 = evaluator.multiply(x, x)
    term1 = evaluator.multiply(term1, x)
    evaluator.multiply_plain_inplace(term1, batch_encoder.encode([quantise(-0.000054479915715, 64)]))

    # 0.1x²
    term2 = seal.Ciphertext()
    term2 = evaluator.multiply(x, x)
    evaluator.multiply_plain_inplace(term2, batch_encoder.encode([quantise(0.1, 64)]))

    # 0.5x
    term3 = seal.Ciphertext()
    term3 = evaluator.multiply_plain(x, batch_encoder.encode([quantise(0.5, 64)]))

    # 0.0001234098040867
    term4 = batch_encoder.encode([quantise(0.24, 64)])

    # add terms
    res = term1
    evaluator.add_inplace(res, term2)
    evaluator.add_inplace(res, term3)
    evaluator.add_plain_inplace(res, term4)

    return res


def swish_approx_3(evaluator, x):
    """
    Deg. 4 approximation
    source: Gottemukkula, 2019
    """
    # -0.1593186187771646x⁴
    term1 = seal.Ciphertext()
    term1 = evaluator.multiply(x, x)
    term1 = evaluator.multiply(term1, term1)
    evaluator.multiply_plain_inplace(term1, batch_encoder.encode([quantise(-0.1593186187771646, 64)]))

    # 0.0000817198735725x³
    term2 = seal.Ciphertext()
    term2 = evaluator.multiply(x, x)
    term2 = evaluator.multiply(term2, x)
    evaluator.multiply_plain_inplace(term2, batch_encoder.encode([quantise(-0.01, 64)]))

    # 0.17x²
    term3 = seal.Ciphertext()
    term3 = evaluator.multiply(x, x)
    evaluator.multiply_plain_inplace(term3, batch_encoder.encode([quantise(0.17, 64)]))

    # 0.5x
    term4 = seal.Ciphertext()
    term4 = evaluator.multiply_plain(x, batch_encoder.encode([quantise(0.5, 64)]))

    # -0.07
    term5 = batch_encoder.encode([quantise(-0.07, 64)])

    # add terms
    res = term1
    evaluator.add_inplace(res, term2)
    evaluator.add_inplace(res, term3)
    evaluator.add_inplace(res, term4)
    evaluator.add_plain_inplace(res, term5)

    return res


def quantise(x, bits):
    # apply simplified quantisation of Wingarz et al.
    n = float(2 ** bits - 1)
    x = np.tanh(x)
    x = np.round(x * n) / n

    # scale to integer range
    return int(np.round(x * n))


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
        # x_res = relu_approx_1(evaluator, x_encrypted)
        # x_res = relu_approx_2(evaluator, x_encrypted)
        # x_res = relu_approx_3(evaluator, x_encrypted)
        # x_res = tanh_approx_1(evaluator, x_encrypted)
        # x_res = tanh_approx_2(evaluator, x_encrypted)
        # x_res = tanh_approx_3(evaluator, x_encrypted)
        # x_res = swish_approx_1(evaluator, x_encrypted)
        # x_res = swish_approx_2(evaluator, x_encrypted)
        # x_res = swish_approx_3(evaluator, x_encrypted)

        # decrypt and decode result
        # evaluator.relinearize_inplace(x_res, relin_keys)
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
    number_of_inputs = 9408  # number of activations in activation layers of MNIST model
    x_rand = np.random.randint(MAX_INT32, MAX_INT64, number_of_inputs, dtype=np.int64)
    print(x_rand)

    # execute and time approximation function
    for x in x_rand:
        process(evaluator, encryptor, decryptor, relin_keys, x)

    end_main = time.time()
    print('SEAL approx. activation function benchmark: end of main function')
    print('time elapsed: ', str(datetime.timedelta(seconds=(end_main - start_main))))
