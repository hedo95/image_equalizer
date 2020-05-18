import numpy as np

def checkShapeIm(im):

    # Si la imagen tiene dimensión NxMx1 es que es monocanal y crearemos el segundo canal multiplicando porsigo mismo
    # Si la imagen tiene un cuarto canal (NxMx4) (brillo por ejemplo) nos lo cargamos
    # Si no es ninguno de esos dos, ni el correcto, lo mandaremos a tomar viento

    shape = im.shape

    if len(shape) is 2:
        im = im[..., np.newaxis]
        shape = im.shape

    elif len(shape) is 3 and shape[2] is 4:
        im = im[..., :3]
        shape = im.shape

    if len(shape) is not 3 or shape[2] not in (1,3):
        raise Exception('La imagen ha de tener uno o tres canales 2D')

    return im

def rgb2bw(_im):

    im = np.array(_im)
    im = checkShapeIm(im)

    if (im.shape[2] == 3):
        im = np.sum(im * [0.299, 0.587, 0.114], axis = 2)
        im = np.stack((im, im, im), axis=2)

    else:
        im = np.sum(im * [0.299, 0.587, 0.114, 0], axis = 2)
        im = np.stack((im, im, im), axis=2)

    return im


def histBW(im):

    im = checkShapeIm(im)

    h = [np.count_nonzero(np.round(im[..., 0] * 255) == gray_level) for gray_level in range(256)]
    # Miramos por cada pixel, a que nivel de grises pertenece

    return np.array(h) / sum(h)

def ecualizaBW(_im):

    im = np.array(_im)
    # Trabajamos con una copia, para que la imagen original no sufra la ecualización
    im = checkShapeIm(im)

    im = (im - np.min(im))/(np.max(im) - np.min(im)) # Normalizamos la imagen

    fIm = histBW(im)
    FIm = np.ndarray(len(fIm))
    FIm[0] = fIm[0]
    for r in range(1, len(fIm)):
        FIm[r] = FIm[r - 1] + fIm[r] # 256 sumas

    FIm /= FIm[-1]
    # Nos garantizamos que el ultimo valor (max) es 1
    # Intentamos que el histograma sea una recta de y = mx lo mas parecido posible para ecualizar el Hist

    minFIm = min(FIm[np.where(FIm > 0)[0]]) # Primer elemento '[0]' donde se da la condición (FIm > 0)

    funcion_densidad = (FIm - minFIm) / (1 - minFIm)

    for m in range(im.shape[0]):
        for n in range(im.shape[1]):
            im[m, n, :] = funcion_densidad[int(np.round(255 * im[m, n, 0]))]

    return im

def yuv2rgb(_im):

    im = np.array(_im)
    im = checkShapeIm(im)

    R = im[...,0] + 1.140 * im[...,2]
    G = im[...,0] - 0.395 * im[...,1] - 0.581 * im[...,2]
    B = im[...,0] + 2.033 * im[...,1]

    im = np.stack((R, G, B), axis = 2)

    return im



def rgb2yuv(_im):

    im = np.array(_im)
    im = checkShapeIm(im)

    # Y = 0.299R + 0.587G + 0.114B
    Y = 0.299 * im[...,0] + 0.587*im[...,1] + 0.114*im[...,2]
    U = 0.492 * (im[...,2] - Y)
    V = 0.877 * (im[..., 0] - Y)

    im = np.stack((Y, U, V), axis = 2)

    return im

def ecualizaYUV(_im):

    im = np.array(_im)
    im = checkShapeIm(im)


    im = rgb2yuv(im)
    im[...,0] = ecualizaBW(im[...,0])[...,0]
    im = yuv2rgb(im)

    return im



