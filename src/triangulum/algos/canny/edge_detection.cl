#line 1

/**
 * Copyright (c) 2016, Nikolay Polyarnyi
 * All rights reserved.
 *
 * This is OpenCL kernels for Canny edge detection.
 */

//Should be defined: W, H
#define u(x, y)             u           [(y + 2) * (W + 2*2) + x + 2]
#define gu(x, y)            gu          [(y + 1) * (W + 2*1) + x + 1]
#define du(x, y)            du          [(y) * W + x]
#define is_extremum(x, y)   is_extremum [(y) * W + x]

__constant float gaussian_kernel[5*5] = {0.01257862,  0.02515723,  0.03144654,  0.02515723,  0.01257862,
                                         0.02515723,  0.05660377,  0.0754717 ,  0.05660377,  0.02515723,
                                         0.03144654,  0.0754717 ,  0.09433962,  0.0754717 ,  0.03144654,
                                         0.02515723,  0.05660377,  0.0754717 ,  0.05660377,  0.02515723,
                                         0.01257862,  0.02515723,  0.03144654,  0.02515723,  0.01257862
};
#define gaussian_kernel(x, y) gaussian_kernel[(y + 2) * 5 + x + 2]

// To convolve with Sobel operator
__kernel void convolve_gaussian(__global const float * u,
                                __global       float * gu) {
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    gu(x, y) = 0;

    for (int dx = -2; dx <= 2; dx++) {
        for (int dy = -2; dy <= 2; dy++) {
            gu(x, y) += u(x + dx, y + dy) * gaussian_kernel(dx, dy);
        }
    }
}


/*
    Sobel operator dx:
        -1, 0, 1
        -2, 0, 2
        -1, 0, 1
*/
__constant float2 sobel_kernel[3*3] = {(float2) (-1.0/8.0, -1.0/8.0), (float2) (0.0, -2.0/8.0),  (float2) (1.0/8.0, -1.0/8.0),
                                       (float2) (-2.0/8.0,  0.0),     (float2) (0.0,  0.0),      (float2) (2.0/8.0,  0.0),
                                       (float2) (-1.0/8.0,  1.0/8.0), (float2) (0.0,  2.0/8.0),  (float2) (1.0/8.0,  1.0/8.0),
};
#define sobel_kernel(x, y) sobel_kernel[(y + 1) * 3 + x + 1]

// To convolve with Sobel operator
__kernel void convolve_sobel(__global const float  * gu,
                             __global       float2 * du) {
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    du(x, y) = (float2) (0, 0);

    if (x == 0 || x == W - 1 || y == 0 || y == H - 1) {
        return;
    }

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            du(x, y) += gu(x + dx, y + dy) * sobel_kernel(dx, dy);
        }
    }
}

__constant int2 dxys[8] = {(int2)  (1, 0),  (int2)  (1,  1),    (int2) (0,  1), (int2) (-1,  1),
                           (int2) (-1, 0),  (int2) (-1, -1),    (int2) (0, -1), (int2) ( 1, -1)};

__kernel void non_maximum_suppression(__global const float  * gu,
                                      __global const float2 * du,
                                      __global       int    * is_extremum) {
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    float anglePi = atan2pi(du(x, y).y, du(x, y).x);
    int direction = (int) round(anglePi / 0.25);
    direction = (direction % 8 + 8) % 8;

    int2 dxy = dxys[direction];
    if (x + dxy.x < 0 || y + dxy.y < 0 || x + dxy.x >= W || y + dxy.y >= H
        || x - dxy.x < 0 || y - dxy.y < 0 || x - dxy.x >= W || y - dxy.y >= H) {
        is_extremum(x, y) = 0;
        return;
    }
    float curDu = length(du(x, y));
    if (curDu > length(du(x - dxy.x, y - dxy.y)) && curDu > length(du(x + dxy.x, y + dxy.y))) {
        is_extremum(x, y) = 1;
    } else {
        is_extremum(x, y) = 0;
    }
}
