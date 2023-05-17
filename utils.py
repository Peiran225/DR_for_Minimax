import numpy as np

def gradient_x(data, label, p, y, alpha, x, beta):
    # beta is not -1 if and only if DR is applied
    pos_index = (1 + label) / 2
    neg_index = (1 - label) / 2
    grad_y = np.zeros_like(y)
    w = y[:-2]
    a, b = y[-2], y[-1]
    coeff1 = 2 * (1 - p) * pos_index * (np.sum(data * w, axis=1) - a)
    coeff2 = 2 * p * neg_index * (np.sum(data * w, axis=1) - b)

    term1 = np.mean(data.T * coeff1, axis=1)
    term2 = np.mean(data.T * coeff2, axis=1)
    term3 = 2 * (1 + alpha) * np.mean(data.T * (p * neg_index - (1 - p) * pos_index), axis=1)

    grad_y[:-2] = term1 + term2 + term3
    grad_y[-2] = - np.mean(coeff1)
    grad_y[-1] = - np.mean(coeff2)
    if beta > 0:
        grad_y = grad_y + (1 / beta) * (y - x)
    
    return grad_y


def gradient_alpha(data, label, p, y, alpha):
    pos_index = (1 + label) / 2
    neg_index = (1 - label) / 2
    w = y[:-2]
    grad_alpha = 2 * np.mean(np.sum(data * w, axis=1) * (p * neg_index - (1 - p) * pos_index)) - 2 * p * (1 - p) * alpha
    return grad_alpha


def dr_subproblem(data, label, p, y, alpha, x, grad_norm, args):
    beta = args.dr_beta
    lr1, lr2 = args.learning_rate[0], args.learning_rate[1]
    bs = args.batch_size
    tolerance = args.dr_tolerance + args.dr_epsilon * grad_norm
    maxiter = args.dr_maxiter

    y_0 = np.copy(y)
    alpha_0 = alpha

    n_sample = data.shape[0]
    oracle = 0
    for iteration in range(maxiter):
        if args.stochastic:
            batch = np.random.randint(0, n_sample, bs)
            grad_y = gradient_x(data[batch, :], label[batch], p, y, alpha, x, beta)
            grad_alpha = gradient_alpha(data[batch, :], label[batch], p, y, alpha)
            oracle += bs
        else:
            grad_y = gradient_x(data, label, p, y, alpha, x, beta)
            grad_alpha = gradient_alpha(data, label, p, y, alpha)
            oracle += n_sample

        y = y - lr1 * grad_y
        alpha = alpha + lr2 * grad_alpha

        grad_norm = np.sqrt(np.sum(grad_y * grad_y) + grad_alpha * grad_alpha)
        if grad_norm <= tolerance:
            break

    distance = np.sqrt(np.sum((y - y_0) * (y - y_0)) + (alpha - alpha_0) * (alpha - alpha_0))
    return y, alpha, oracle, grad_norm, distance


def dr_subproblem2(data, label, p, y, alpha, v_y, v_alpha, x, grad_norm, args):
    beta = args.dr_beta
    lr1, lr2 = args.learning_rate[0], args.learning_rate[1]
    bs = args.batch_size
    tolerance = args.dr_tolerance + args.dr_epsilon * grad_norm
    maxiter = args.dr_maxiter

    y_0 = np.copy(y)
    alpha_0 = alpha

    theta = 0.9
    rho = 0.01

    n_sample = data.shape[0]
    oracle = 0
    for iteration in range(maxiter):
        if args.stochastic:
            batch = np.random.randint(0, n_sample, bs)
            grad_y = gradient_x(data[batch, :], label[batch], p, y, alpha, x, beta)
            grad_alpha = gradient_alpha(data[batch, :], label[batch], p, y, alpha)
            v_y = theta * v_y + (1 - theta) * grad_y * grad_y
            v_alpha = theta * v_alpha + (1 - theta) * grad_alpha * grad_alpha
            oracle += bs
        else:
            grad_y = gradient_x(data, label, p, y, alpha, x, beta)
            grad_alpha = gradient_alpha(data, label, p, y, alpha)
            oracle += n_sample

        y = y - lr1 * grad_y / (np.sqrt(v_y) + rho)
        alpha = alpha + lr2 * grad_alpha / (np.sqrt(v_alpha) + rho)

        grad_norm = np.sqrt(np.sum(grad_y * grad_y) + grad_alpha * grad_alpha)
        if grad_norm <= tolerance:
            break

    distance = np.sqrt(np.sum((y - y_0) * (y - y_0)) + (alpha - alpha_0) * (alpha - alpha_0))
    return y, alpha, v_y, v_alpha, oracle, grad_norm, distance

