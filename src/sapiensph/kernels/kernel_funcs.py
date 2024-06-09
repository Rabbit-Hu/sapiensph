import warp as wp


@wp.func
def sph_W(r: wp.vec3, h: float):
    q = wp.length(r) / h
    if q < 1.0:
        return 1.0 / (wp.pi * h ** 3.0) * (1.0 - 1.5 * q ** 2.0 + 0.75 * q ** 3.0)
    elif q < 2.0:
        return 1.0 / (wp.pi * h ** 3.0) * 0.25 * (2.0 - q) ** 3.0
    else:
        return 0.0


@wp.func
def sph_dWdr(r: wp.vec3, h: float):
    q = wp.length(r) / h
    if q < 1.0:
        return 1.0 / (wp.pi * h ** 4.0) * (-3.0 * q + 2.25 * q ** 2.0) * wp.normalize(r)
    elif q < 2.0:
        return 1.0 / (wp.pi * h ** 4.0) * -0.75 * (2.0 - q) ** 2.0 * wp.normalize(r)
    else:
        return wp.vec3(0.0, 0.0, 0.0)
