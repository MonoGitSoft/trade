import numpy as np

def lin_reg(data):
    x = np.array(range(data.size), dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    lin_reg_result = np.linalg.lstsq(A, data, rcond=None)
    result = {'gradient':lin_reg_result[0][0], 'residual':lin_reg_result[1], 'norm_residual': lin_reg_result[1]/data.size}
    return result


def calc_sma_seq(closeMid,window_sizes):
    data_sma = np.zeros((len(closeMid), len(window_sizes)), dtype=float)
    column = 0
    for win_size in window_sizes:
        result = slide_window_filter(closeMid, win_size)
        data_sma[:, column] = result
        column = column + 1

    column = 0
    max_column = len(window_sizes) - 1
    tmp = np.zeros((len(closeMid), max_column), dtype=float)
    tmp_counter = 0
    print(max_column)
    for i in range(max_column):
        tmp[:, tmp_counter] = np.subtract(data_sma[:, i], data_sma[:, i + 1])
        tmp[:, tmp_counter] = tmp[:, tmp_counter] * 1 / np.sqrt(np.var(tmp[:, tmp_counter]))
        a = tmp[:, tmp_counter]
        print(np.var(a))
        tmp_counter = tmp_counter + 1
    data_sma = np.copy(tmp)
    return data_sma

def calc_gradients(closeMid, window_sizes):
    data_gradients = np.zeros((len(closeMid),len(window_sizes)), dtype=float)
    column = 0
    for win_size in window_sizes:
        result = gradient_linreg_slidewindow(closeMid, win_size)
        data_gradients[:,column] = result['gradiens']
        data_gradients[:,column] = data_gradients[:,column] * 1 / np.sqrt(np.var(data_gradients[:,column]))
        column = column + 1
    return data_gradients

def gradient_linreg_slidewindow(data, window_size):
    window = np.array(np.zeros(np.int(window_size), dtype=float))
    window[:] = data[0]
    count = 0
    gradiens = list(range(data.size))
    deviations = list(range(data.size))
    for x in range(data.size):
        window[count] = data[x]
        result = lin_reg(window)
        gradiens[x] = float(result['gradient'])
        #deviations[x] = float(result['norm_residual'])
        count = count + 1
        if count == window_size:
            count = 0
    result = {'gradiens': np.array(gradiens)}
    return  result

def slide_window_filter(data, window_size):
    window = np.array(np.zeros(np.int(window_size), dtype=float))
    window[:] = data[0]
    filtered = list(range(data.size))
    count = 0
    for x in range(data.size):
        window[count] =  data[x]
        filtered[x] = np.mean(window)
        count = count + 1
        if count == window_size:
            count = 0
    return filtered