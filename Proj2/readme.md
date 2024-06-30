## 说明

### 1. Harris角点检测器

Harris角点检测器使用一个二元函数来描述当图像沿着任意方向 $(u, v)$ 进行平移时的像素强度差分，其表达式如下：

```math
E(u, v) = \sum_{x, y}w(x, y)[I(x+u, y+v) - I(x, y)]^2
```

其中 $w(x, y)$ 是窗口函数，它可以是矩形窗口，窗口内的取值全部为1，也可以是高斯窗口，窗口内的取值为二元高斯函数的函数值。显然， $E(0, 0)$ 是 $E(u, v)$ 的局部极小值点，将 $I(x+u, y+v)$ 在点 $(x, y)$ 处进行一阶泰勒展开，可以得到：

```math
E(u, v) = \sum_{x, y}w(x, y)[I(x+u, y+v) - I(x, y)]^2 \\
\approx \sum_{x, y}w(x, y)[I(x, y) + I_x u + I_y v - I(x, y)]^2 \\
=\sum_{x, y}w(x, y)[I_x u + I_y v]^2 \\
= \sum_{x, y}[(I_x u)^2 + (I_y v)^2 + 2(I_x I_y uv)] \\
= \sum_{x, y}w(x, y)
\left[\begin{array}{cc}u &v\end{array}\right]
\left[\begin{array}{cc}I_x I_x &I_x I_y \\ I_x I_y &I_y I_y\end{array}\right]
\left[\begin{array}{c}u \\ v\end{array}\right] \\
= \left[\begin{array}{cc}u &v\end{array}\right]
\sum_{x, y}w(x, y)
\left[\begin{array}{cc}I_x I_x &I_x I_y \\ I_x I_y &I_y I_y\end{array}\right]
\left[\begin{array}{c}u \\ v\end{array}\right]
```

令：

```math
M = \sum_{x, y}w(x, y)
\left[\begin{array}{cc}I_x I_x &I_x I_y \\ I_x I_y &I_y I_y\end{array}\right]
```

 则

```math
E(u, v) \approx \left[\begin{array}{cc}u &v\end{array}\right] M 
\left[\begin{array}{c}u \\ v\end{array}\right]
```

其中 $I_x$ 和 $I_y$ 分别是图像在沿着 $x$ 方向和 $y$ 方向上的偏导数。

紧接着根据矩阵 $M$ 构造一个响应函数 $R$ ，其方程为：

```math
R = \det(M) - k[\mathrm{tr}(M)^2]
```

其中：

* $\det(M) = \lambda_1 \cdot \lambda_2$；
* $\mathrm{tr}(M) = \lambda_1 + \lambda_2$；
* $\lambda_1$和 $\lambda_2$ 分别是矩阵 $M$ 的两个特征值。

因此 $R$ 是 $\lambda_1$ 和 $\lambda_2$ 的函数，它们共同决定了一个窗口区域是平面、边缘还是角点：

* 当 $|R|$ 很小时，则 $\lambda_1$ 和 $\lambda_2$ 也很小，此时该区域是平面；
* 当 $R < 0$ 时，则有 $\lambda_1 \gg \lambda_2$ 或 $\lambda_1 \ll \lambda_2$ ，此时该区域是边缘；
* 当 $R$ 很大且 $\lambda_1$ 与 $\lambda_2$ 数值大小相当，即 $\lambda_1 \sim \lambda_2$ 时，此时该区域是角点。

```python
def get_interest_points(image, feature_width):
    '''
    Returns a set of interest points for the input image

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # Compute $I_x^2$, $I_y^2$, $I_x * I_y$
    partial_x = filters.sobel_h(image)
    partial_x2 = np.square(partial_x)
    partial_y = filters.sobel_v(image)
    partial_y2 = np.square(partial_y)
    partial_xy = partial_x * partial_y

    # Apply Gaussian filter, equivalent to compute elements of $M$
    sigma = 1.0
    partial_x2 = filters.gaussian(partial_x2, sigma=sigma)
    partial_y2 = filters.gaussian(partial_y2, sigma=sigma)
    partial_xy = filters.gaussian(partial_xy, sigma=sigma)

    if len(image.shape) == 2:
        h, w = image.shape
    else:
        h, w, _ = image.shape

    # Construct matrix M in all location
    M = np.zeros(h * w * 4, np.float32)
    M[0::4] = partial_x2.flatten()
    M[1::4] = partial_xy.flatten()
    M[2::4] = partial_xy.flatten()
    M[3::4] = partial_y2.flatten()
    M = M.reshape((-1, 2, 2))

    det_M = np.linalg.det(M)
    tr_M = np.trace(M, axis1=1, axis2=2)

    # Parameters used in harris corner detector
    k = 0.04
    c = 0.02

    # Compute cornerness score
    cornerness = det_M - k * np.square(tr_M)

    # For $|R| < \epsilon$, used to remove flat area
    cornerness[cornerness < c * cornerness.max()] = 0

    # Recover shape
    cornerness = cornerness.reshape((h, w))

    # Non max suppression
    coordinates = feature.peak_local_max(cornerness, min_distance=15)

    # Be aware of the index for xs and ys
    xs = coordinates[:, 1]
    ys = coordinates[:, 0]

    return xs, ys
```





### 2. 我自己的另一种推导方式

将 $E(u, v)$ 在 $(0, 0)$ 点进行二阶泰勒展开代替 $I(x+u, y+v)$ 在 $(x, y)$ 点处的一阶泰勒展开：

```math
E(u, v) = \sum_{x, y}w(x, y)[I(x+u, y+v) - I(x, y)]^2 \\
\approx E(0, 0) + 
\left[\begin{array}{cc}u &v\end{array}\right] 
\left[\begin{array}{c}E_u(0,0) \\ E_v(0, 0) \end{array}\right] + 
\frac{1}{2} 
\left[\begin{array}{cc}u &v\end{array}\right] 
\left[\begin{array}{cc}E_{uu}(0, 0) &E_{uv}(0, 0) \\ E_{vu}(0, 0) &E_{vv}(0, 0)\end{array}\right]
\left[\begin{array}{c}u \\ v\end{array}\right] \\
= \frac{1}{2} 
\left[\begin{array}{cc}u &v\end{array}\right] 
\left[\begin{array}{cc}E_{uu}(0, 0) &E_{uv}(0, 0) \\ E_{vu}(0, 0) &E_{vv}(0, 0)\end{array}\right]
\left[\begin{array}{c}u \\ v\end{array}\right] \\
= \left[\begin{array}{cc}u &v\end{array}\right]
\sum_{x, y}w(x, y)
\left[\begin{array}{cc}I_{xx}(x, y) &I_{xy}(x, y) \\ I_{yx}(x, y) &I_{yy}(x, y)\end{array}\right]
\left[\begin{array}{c}u \\ v\end{array}\right]
```

令

```math
M = \sum_{x, y}w(x, y)
\left[\begin{array}{cc}I_{xx}(x, y) &I_{xy}(x, y) \\ I_{yx}(x, y) &I_{yy}(x, y)\end{array}\right]
```

则

```math
E(u, v) \approx \left[\begin{array}{cc}u &v\end{array}\right] M 
\left[\begin{array}{c}u \\ v\end{array}\right]
```

其中 $I_{xx}$ 和 $I_{yy}$ 分别是图像在沿着 $x$ 方向和 $y$ 方向上的二阶偏导数， $I_{xy}$ 和 $I_{yx}$ 分别是图像的二阶混合偏导数。

```python
def get_interest_points(image, feature_width):
    '''
    Returns a set of interest points for the input image

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # Compute $I_x^2$, $I_y^2$, $I_{xy}$, $I_{yx}$
    partial_x = filters.sobel_h(image)
    partial_x2 = filters.sobel_h(partial_x)
    partial_y = filters.sobel_v(image)
    partial_y2 = filters.sobel_v(partial_y)
    partial_xy = filters.sobel_v(partial_x)
    partial_yx = filters.sobel_h(partial_y)


    # Apply Gaussian filter, equivalent to compute elements of $M$
    sigma = 1.0
    partial_x2 = filters.gaussian(partial_x2, sigma=sigma)
    partial_y2 = filters.gaussian(partial_y2, sigma=sigma)
    partial_xy = filters.gaussian(partial_xy, sigma=sigma)
    partial_yx = filters.gaussian(partial_yx, sigma=sigma)

    if len(image.shape) == 2:
        h, w = image.shape
    else:
        h, w, _ = image.shape

    # Construct matrix M in all location
    M = np.zeros(h * w * 4, np.float32)
    M[0::4] = partial_x2.flatten()
    M[1::4] = partial_xy.flatten()
    M[2::4] = partial_yx.flatten()
    M[3::4] = partial_y2.flatten()
    M = M.reshape((-1, 2, 2))

    det_M = np.linalg.det(M)
    tr_M = np.trace(M, axis1=1, axis2=2)

    # Parameters used in harris corner detector
    k = 0.04
    c = 0.02

    # Compute cornerness score
    cornerness = det_M - k * np.square(tr_M)

    # For $|R| < \epsilon$, used to remove flat area
    cornerness[cornerness < c * cornerness.max()] = 0

    # Recover shape
    cornerness = cornerness.reshape((h, w))

    # Non max suppression
    coordinates = feature.peak_local_max(cornerness, min_distance=15)

    # Be aware of the index for xs and ys
    xs = coordinates[:, 1]
    ys = coordinates[:, 0]

    return xs, ys
```





## 参考文献

[1] *Understanding features*. OpenCV. (n.d.). Retrieved April 25, 2022, from https://docs.opencv.org/4.x/df/d54/tutorial_py_features_meaning.html 

[2] *Introduction to SIFT (scale-invariant feature transform)*. OpenCV. (n.d.). Retrieved April 25, 2022, from https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html 

