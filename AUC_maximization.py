import argparse
import numpy as np
from dataset import libsvm_loader
from utils import gradient_x, gradient_alpha, dr_subproblem, dr_subproblem2
import time 

# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='AUC Maximization Problem')
parser.add_argument('--dataset', default='a9a', type=str, help='name of dataset')
parser.add_argument('--reg_lambda', default=0.001, type=float, help='coefficient of regularization')
parser.add_argument('--algorithm', default='SPAUC', type=str, help='name of algorithm')
parser.add_argument('--num_epochs', default=500, type=int, help='number of epochs to train')
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--learning_rate', default=[0.01,0.1], nargs='+', type=float, help='learning rate')
parser.add_argument('--dr_beta', default=0.1, type=float, help='hyper-parameter of DR algorithm')
parser.add_argument('--dr_epsilon', default=0.95, type=float, help='hyper-parameter of DR algorithm')
parser.add_argument('--dr_tolerance', default=0.15, type=float, help='hyper-parameter of DR algorithm')
parser.add_argument('--dr_maxiter', default=20, type=int, help='hyper-parameter of DR algorithm')
parser.add_argument('--mda_beta', default=0.1, type=float, help='hyper-parameter of MDA algorithm')
parser.add_argument('--mda_rho', default=0.01, type=float, help='hyper-parameter of MDA algorithm')
parser.add_argument('--alternative', action='store_true', help='using alternative update rule')
parser.add_argument('--stochastic', action='store_true', help='using stochastic gradient oracle')
parser.add_argument('--variance_reduce', action='store_true', help='using VR gradient oracle')
parser.add_argument('--mda_q', default=1000, type=int, help='hyper-parameter of MDA algorithm')
parser.add_argument('--ppa_gamma', default=0.1, type=float, help='hyper-parameter of DR algorithm')
parser.add_argument('--ppa_maxiter', default=20, type=int, help='hyper-parameter of DR algorithm')
parser.add_argument('--print_freq', default=10, type=int, help='frequency to print train stats')
parser.add_argument('--out_fname', default='result.csv', type=str, help='name of output file')
# --------------------------------------------------------------------------- #


def get_AUC(data, label, w):
    num_pos, num_neg, num_miss = 0, 0, 0
    L = []
    for idx in range(data.shape[0]):
        if label[idx] == 1:
            L.append((np.inner(data[idx, :], w), 0))
            num_pos += 1
        else:
            L.append((np.inner(data[idx, :], w), 1))
            num_neg += 1

    num_pair = num_pos * num_neg
    L_sort = sorted(L, reverse=True)

    s = 0
    for item in L_sort:
        if item[1] == 0:
            num_miss += s
        s += item[1]
    return 1 - (num_miss / num_pair)


def get_val(data, label, w, a, b, alpha, args):
    n_sample = data.shape[0]
    num_pos = sum(label == 1)
    p = num_pos / n_sample

    pos_index = (1 + label) / 2
    neg_index = (1 - label) / 2

    linear = np.sum(data * w, axis=1)
    term1 = (1 - p) * np.mean(pos_index * (linear - a) * (linear - a))
    term2 = p * np.mean(neg_index * (linear - b) * (linear - b))
    term3 = 2 * (1 + alpha) * np.mean(linear * (p * neg_index - (1 - p) * pos_index))
    fval = term1 + term2 + term3 + p * (1 - p) * (1 - alpha * alpha) + args.reg_lambda * sum(np.abs(w))
    return fval


def SPAUC(data, label, args):
    reg_lambda = args.reg_lambda
    T = args.num_epochs
    bs = args.batch_size
    lr = args.learning_rate[0]
    freq = args.print_freq
    out_fname = args.out_fname

    n_sample = data.shape[0]
    w = np.zeros(data.shape[1])
    n_pos, n_neg, n_total = 0, 0, 0
    s_pos = np.zeros(data.shape[1])
    s_neg = np.zeros(data.shape[1])

    elapsed_time = 0.0
    oracle = 0

    with open(out_fname, 'w') as f:
        f.write('iteration,time,oracle,fval,AUC\n')
        f.write('0,0.00,0,0.0000,0.0000\n')

    for iteration in range(T):
        t_begin = time.time()
        batch = np.random.randint(0, n_sample, bs)
        oracle += bs
        n_total += bs
        n_pos += sum(label[batch] == 1)
        n_neg += sum(label[batch] == -1)

        pos_index = (1 + label[batch]) / 2
        neg_index = (1 - label[batch]) / 2

        s_pos = s_pos + np.sum(data[batch, :].T * pos_index, axis=1)
        s_neg = s_neg + np.sum(data[batch, :].T * neg_index, axis=1)

        u = s_pos / n_pos if n_pos > 0 else np.zeros_like(s_pos)
        v = s_neg / n_neg if n_neg > 0 else np.zeros_like(s_neg)
        p = n_pos / n_total

        grad = 2 * p * (1 - p) * (v - u) + 2 * p * (1 - p) * np.inner(w, v - u) * (v - u)
        central_u = data[batch, :] - u
        central_v = data[batch, :] - v
        term1 = 2 * (1 - p) * np.mean(central_u.T * pos_index * np.sum(central_u * w, axis=1), axis=1)
        term2 = 2 * p * np.mean(central_v.T * neg_index * np.sum(central_v * w, axis=1), axis=1)
        grad = grad + term1 + term2

        w = np.sign(w - lr * grad) * np.maximum(np.abs(w - lr * grad) - lr * reg_lambda, 0)

        t_end = time.time()
        elapsed_time += (t_end - t_begin)

        if (iteration + 1) % freq == 0:
            a = np.inner(w, u)
            b = np.inner(w, v)
            alpha = b - a
            AUC = get_AUC(data, label, w)
            fval = get_val(data, label, w, a, b, alpha, args)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%d,%.4f,%.4f\n' % (iteration + 1, elapsed_time, oracle, fval, AUC))


def SPAM(data, label, args):
    reg_lambda = args.reg_lambda
    T = args.num_epochs
    bs = args.batch_size
    lr = args.learning_rate[0]
    freq = args.print_freq
    out_fname = args.out_fname

    n_sample = data.shape[0]
    w = np.zeros(data.shape[1])
    n_pos, n_neg, n_total = 0, 0, 0
    s_pos = np.zeros(data.shape[1])
    s_neg = np.zeros(data.shape[1])

    elapsed_time = 0.0
    oracle = 0

    with open(out_fname, 'w') as f:
        f.write('iteration,time,oracle,fval,AUC\n')
        f.write('0,0.00,0,0.0000,0.0000\n')
    
    for iteration in range(T):
        t_begin = time.time()
        batch = np.random.randint(0, n_sample, bs)
        oracle += bs
        n_total += bs
        n_pos += sum(label[batch] == 1)
        n_neg += sum(label[batch] == -1)

        pos_index = (1 + label[batch]) / 2
        neg_index = (1 - label[batch]) / 2

        s_pos = s_pos + np.sum(data[batch, :].T * pos_index, axis=1)
        s_neg = s_neg + np.sum(data[batch, :].T * neg_index, axis=1)

        u = s_pos / n_pos if n_pos > 0 else np.zeros_like(s_pos)
        v = s_neg / n_neg if n_neg > 0 else np.zeros_like(s_neg)
        p = n_pos / n_total

        central_u = data[batch, :] - u
        central_v = data[batch, :] - v
        term1 = 2 * (1 - p) * np.mean(data[batch, :].T * pos_index * np.sum(central_u * w, axis=1), axis=1)
        term2 = 2 * p * np.mean(data[batch, :].T * neg_index * np.sum(central_v * w, axis=1), axis=1)
        term3 = 2 * p * (1 - p) * (1 + np.inner(w, v - u)) * (v - u) 
        grad = term1 + term2 + term3

        w = np.sign(w - lr * grad) * np.maximum(np.abs(w - lr * grad) - lr * reg_lambda, 0)

        t_end = time.time()
        elapsed_time += (t_end - t_begin)

        if (iteration + 1) % freq == 0:
            a = np.inner(w, u)
            b = np.inner(w, v)
            alpha = b - a
            AUC = get_AUC(data, label, w)
            fval = get_val(data, label, w, a, b, alpha, args)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%d,%.4f,%.4f\n' % (iteration + 1, elapsed_time, oracle, fval, AUC))


def PAGDA(data, label, args):
    reg_lambda = args.reg_lambda
    T = args.num_epochs
    lr1, lr2 = args.learning_rate[0], args.learning_rate[1]
    bs = args.batch_size
    freq = args.print_freq
    out_fname = args.out_fname

    n_sample = data.shape[0]
    num_pos = sum(label == 1)
    p = num_pos / n_sample

    x = np.zeros(data.shape[1] + 2)  # w, a, b
    alpha = 0

    elapsed_time = 0.0
    oracle = 0

    with open(out_fname, 'w') as f:
        f.write('iteration,time,oracle,fval,AUC\n')
        f.write('0,0.00,0,0.0000,0.0000\n')

    for iteration in range(T):
        t_begin = time.time()

        batch = np.random.randint(0, n_sample, bs)
        grad_x = gradient_x(data[batch, :], label[batch], p, x, alpha, x, -1)
        if args.alternative:
            x = x - lr1 * grad_x
            x[:-2] = np.sign(x[:-2]) * np.maximum(np.abs(x[:-2]) - lr1 * reg_lambda, 0)
            grad_alpha = gradient_alpha(data[batch, :], label[batch], p, x, alpha)
        else:
            grad_alpha = gradient_alpha(data[batch, :], label[batch], p, x, alpha)
            x = x - lr1 * grad_x
            x[:-2] = np.sign(x[:-2]) * np.maximum(np.abs(x[:-2]) - lr1 * reg_lambda, 0)
        alpha = alpha + lr2 * grad_alpha

        oracle += bs
        t_end = time.time()
        elapsed_time += (t_end - t_begin)

        if (iteration + 1) % freq == 0:
            w = x[:-2]
            a, b = x[-2], x[-1]
            AUC = get_AUC(data, label, w)
            fval = get_val(data, label, w, a, b, alpha, args)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%d,%.4f,%.4f\n' % (iteration + 1, elapsed_time, oracle, fval, AUC))


def MDA(data, label, args):
    reg_lambda = args.reg_lambda
    T = args.num_epochs
    lr1, lr2 = args.learning_rate[0], args.learning_rate[1]
    bs = args.batch_size
    beta = args.mda_beta
    rho = args.mda_rho
    q = args.mda_q
    freq = args.print_freq
    out_fname = args.out_fname

    n_sample = data.shape[0]
    num_pos = sum(label == 1)
    p = num_pos / n_sample

    x = np.zeros(data.shape[1] + 2)  # w, a, b
    alpha = 0
    x_old = np.copy(x)
    alpha_old = alpha
    v_x = np.zeros_like(x)
    v_alpha = 0

    elapsed_time = 0.0
    oracle = 0

    with open(out_fname, 'w') as f:
        f.write('iteration,time,oracle,fval,AUC\n')
        f.write('0,0.00,0,0.0000,0.0000\n')

    for iteration in range(T):
        t_begin = time.time()

        if (not args.stochastic) or (args.variance_reduce and iteration % q == 0):
            batch = np.random.randint(0, n_sample, 20000)
            grad_x = gradient_x(data[batch, :], label[batch], p, x, alpha, x, -1)
            grad_alpha = gradient_alpha(data[batch, :], label[batch], p, x, alpha)
            # grad_x = gradient_x(data, label, p, x, alpha, x, -1)
            # grad_alpha = gradient_alpha(data, label, p, x, alpha)
            v_x = beta * v_x + (1 - beta) * grad_x * grad_x
            v_alpha = beta * v_alpha + (1 - beta) * grad_alpha * grad_alpha
            # oracle += n_sample
            oracle += 20000
        elif args.variance_reduce:
            batch = np.random.randint(0, n_sample, bs)
            grad_x_new = gradient_x(data[batch, :], label[batch], p, x, alpha, x, -1)
            grad_alpha_new = gradient_alpha(data[batch, :], label[batch], p, x, alpha)
            v_x = beta * v_x + (1 - beta) * grad_x_new * grad_x_new
            v_alpha = beta * v_alpha + (1 - beta) * grad_alpha_new * grad_alpha_new
            grad_x_old = gradient_x(data[batch, :], label[batch], p, x_old, alpha_old, x_old, -1)
            grad_alpha_old = gradient_alpha(data[batch, :], label[batch], p, x_old, alpha_old)
            grad_x = grad_x + grad_x_new - grad_x_old
            grad_alpha = grad_alpha + grad_alpha_new - grad_alpha_old
            oracle += 2 * bs
        else:
            batch = np.random.randint(0, n_sample, bs)
            grad_x = gradient_x(data[batch, :], label[batch], p, x, alpha, x, -1)
            grad_alpha = gradient_alpha(data[batch, :], label[batch], p, x, alpha)
            v_x = beta * v_x + (1 - beta) * grad_x * grad_x
            v_alpha = beta * v_alpha + (1 - beta) * grad_alpha * grad_alpha
            oracle += bs

        delta_x = lr1 * grad_x / (np.sqrt(v_x) + rho)
        if args.variance_reduce:
            x_old = np.copy(x)
            alpha_old = alpha
        x = x - delta_x
        x[:-2] = np.sign(x[:-2]) * np.maximum(np.abs(x[:-2]) - lr1 * reg_lambda / (np.sqrt(v_x[:-2]) + rho), 0)
        alpha = alpha + lr2 * grad_alpha / (np.sqrt(v_alpha) + rho)

        t_end = time.time()
        elapsed_time += (t_end - t_begin)

        if (iteration + 1) % freq == 0:
            w = x[:-2]
            a, b = x[-2], x[-1]
            AUC = get_AUC(data, label, w)
            fval = get_val(data, label, w, a, b, alpha, args)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%d,%.4f,%.4f\n' % (iteration + 1, elapsed_time, oracle, fval, AUC))


def DRAUC(data, label, args):
    reg_lambda = args.reg_lambda
    T = args.num_epochs
    beta = args.dr_beta
    freq = args.print_freq
    out_fname = args.out_fname

    n_sample = data.shape[0]
    num_pos = sum(label == 1)
    p = num_pos / n_sample

    x = np.zeros(data.shape[1] + 2)  # w, a, b
    y = np.copy(x)
    z = np.copy(x)
    alpha = 0

    v_y = np.zeros_like(y)
    v_alpha = 0

    elapsed_time = 0.0
    oracle = 0
    norm_min, dist_min = -1, -1
    grad_norm = 0.2
    
    with open(out_fname, 'w') as f:
        f.write('iteration,time,oracle,fval,AUC\n')
        f.write('0,0.00,0,0.0000,0.0000\n')

    for iteration in range(T):
        t_begin = time.time()

        x = x + z - y
        # y, alpha, oracle_add, grad_norm, distance = dr_subproblem(data, label, p, y, alpha, x, grad_norm, args)
        y, alpha, v_y, v_alpha, oracle_add, grad_norm, distance = dr_subproblem2(data, label, p, y, alpha, v_y, v_alpha, x, grad_norm, args)
        z = 2 * y - x
        z[:-2] = np.sign(z[:-2]) * np.maximum(np.abs(z[:-2]) - reg_lambda * beta, 0)

        oracle += oracle_add

        t_end = time.time()
        elapsed_time += (t_end - t_begin)

        norm_min = grad_norm if norm_min < 0 else min(norm_min, grad_norm)
        dist_min = distance if dist_min < 0 else min(dist_min, distance)

        if (iteration + 1) % freq == 0:
            w = x[:-2]
            a, b = x[-2], x[-1]
            AUC = get_AUC(data, label, w)
            fval = get_val(data, label, w, a, b, alpha, args)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%d,%.4f,%.4f\n' % (iteration + 1, elapsed_time, oracle, fval, AUC))

    print('gradient norm: %f, moving distance: %f' % (norm_min, dist_min))


def PPA(data, label, args):
    reg_lambda = args.reg_lambda
    T = args.num_epochs
    gamma = args.ppa_gamma
    maxiter = args.ppa_maxiter
    lr1, lr2 = args.learning_rate[0], args.learning_rate[1]
    bs = args.batch_size
    freq = args.print_freq
    out_fname = args.out_fname

    n_sample = data.shape[0]
    num_pos = sum(label == 1)
    p = num_pos / n_sample

    x = np.zeros(data.shape[1] + 2)  # w, a, b
    alpha = 0

    elapsed_time = 0.0
    oracle = 0

    with open(out_fname, 'w') as f:
        f.write('iteration,time,oracle,fval,AUC\n')
        f.write('0,0.00,0,0.0000,0.0000\n')

    for iteration in range(T):
        t_begin = time.time()
        x0 = np.copy(x)
        alpha0 = alpha
        for loop in range(maxiter):
            batch = np.random.randint(0, n_sample, bs)
            grad_x = gradient_x(data[batch, :], label[batch], p, x, alpha, x, -1) + gamma * (x - x0)
            grad_alpha = gradient_alpha(data[batch, :], label[batch], p, x, alpha) - gamma * (alpha - alpha0)
            x = x - lr1 * grad_x
            x[:-2] = np.sign(x[:-2]) * np.maximum(np.abs(x[:-2]) - lr1 * reg_lambda, 0)
            alpha = alpha + lr2 * grad_alpha
            oracle += bs

        t_end = time.time()
        elapsed_time += (t_end - t_begin)

        if (iteration + 1) % freq == 0:
            w = x[:-2]
            a, b = x[-2], x[-1]
            AUC = get_AUC(data, label, w)
            fval = get_val(data, label, w, a, b, alpha, args)
            with open(out_fname, '+a') as f:
                f.write('%d,%.2f,%d,%.4f,%.4f\n' % (iteration + 1, elapsed_time, oracle, fval, AUC))


def main():
    args = parser.parse_args()
    data, label = libsvm_loader(args.dataset)

    if args.algorithm == 'DRAUC':
        DRAUC(data, label, args)
    elif args.algorithm == 'SPAUC':
        SPAUC(data, label, args)
    elif args.algorithm == 'SPAM':
        SPAM(data, label, args)
    elif args.algorithm == 'PAGDA':
        PAGDA(data, label, args)
    elif args.algorithm == 'MDA':
        MDA(data, label, args)
    elif args.algorithm == 'PPA':
        PPA(data, label, args)
    else:
        raise NotImplementedError('This method is not implemented!')


if __name__ == '__main__':
    main()

