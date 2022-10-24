def get_sd_mean(data):
    sd = np.std(data, axis=0, dtype=float)
    mean = np.mean(data, axis=0, dtype=float)
    return sd, mean


def standard_data(data, sd, mean, kind="all"):
    std_data = data

    if data.shape.__len__() == 1:
        for d in range(0, data.shape[0]):
            calc = (data[d] - mean[d]) / sd[d]
            std_data[d] = float(calc)
        return std_data

    for p in range(0, data.shape[0]):
        if kind == "non-error":
            calc = (data[p][0] - mean[0]) / sd[0]
            std_data[p][0] = float(calc)
        else:
            for d in range(0, data[0].shape[0]):
                calc = (data[p][d] - mean[d]) / sd[d]
                std_data[p][d] = float(calc)

    return std_data
