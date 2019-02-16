import numpy as np

def lin_reg(data):
    x = np.array(range(data.size), dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    lin_reg_result = np.linalg.lstsq(A, data, rcond=None)
    result = {'gradient':lin_reg_result[0][0], 'residual':lin_reg_result[1], 'norm_residual': lin_reg_result[1]/data.size}
    return result

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
        deviations[x] = float(result['norm_residual'])
        count = count + 1
        if count == window_size:
            count = 0
    result = {'gradiens': np.array(gradiens),'deviations':np.array(deviations)}
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