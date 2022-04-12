import numpy as np

def get_lines(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = lines[:-1]

    lines = [float(line.strip()) for line in lines]
    lines = np.array(lines)
    return lines

# filename = 'log_err_CutMix.txt'
# filename = 'log_err_LS.txt'
# filename = 'log_err_Drop.txt'
filename = 'log_err_Ensemble.txt'
lines = get_lines(filename)
mean = np.mean(lines)
std = np.std(lines)
print('mean: ', mean, 'std:', std)
