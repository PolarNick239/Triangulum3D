#line 1

/**
 * Copyright (c) 2016, Nikolay Polyarnyi
 * All rights reserved.
 *
 * This is OpenCL kernels for central line extraction from multiple stripes.
 * The main idea is to calculate distance to stripe border,
 * than apply non-maximum suppression on distance image (like in Canny edge detector).
 */

//Should be defined: W, H


#define type(x, y) type[(y) * W + x]
#define is_edge_pixel(x, y) is_edge_pixel[(y) * W + x]

__kernel void detect_edge_pixels(__global const int * type,
                                 __global       int * is_edge_pixel) {
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    int same_number = 0;
    int dx_sum = 0;
    int dy_sum = 0;
    int this_type = type(x, y);
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (type(x + dx, y + dy) == this_type) {
                same_number += 1;
                dx_sum += dx;
                dy_sum += dy;
            }
        }
    }
    same_number -= 1;  // Not count our-self

    int total_neib_pixels = 3 * 3 - 1;
    int threshold = 3;
    if (threshold <= same_number && same_number <= (total_neib_pixels - threshold)
        && length((float2) (dx_sum / (1.0f * same_number), dy_sum / (1.0f * same_number))) >= 0.3) {
        is_edge_pixel(x, y) = 1;
    } else {
        is_edge_pixel(x, y) = 0;
    }
}


#define PLUS_INFIMUM (W + H + 239.0f)

#define distance(x, y) distance[(y) * W + x]
#define is_maximum(x, y) is_maximum[(y) * W + x]

__kernel void nearest_edge_iter(__global const int * type,
                                __global const int * is_edge_pixel,
                                __global       float * distance,
                                __global       int * changed) {
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);
    if (is_edge_pixel(x, y)) {
        distance(x, y) = 0.0f;
        return;
    }

    int this_type = type(x, y);
    float min_distance = PLUS_INFIMUM;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (type(x + dx, y + dy) == this_type) {
                min_distance = min(min_distance, 1.0f + distance(x + dx, y + dy));
                // TODO: try not 1.0f+, but length(dx, dy)+
            }
        }
    }

    if (min_distance < distance(x, y)) {
        distance(x, y) = min_distance;
        changed[0] = 1;
    }
}


#define u(x, y) u[(y) * W + x]
#define kern(x, y) kern[(y) * 3 + x]
#define res(x, y) res[(y) * W + x]

// To convolve with Sobel operator
__kernel void convolve(__global const float  * u,
                       __global const float2 * kern,  // TODO: make it local
                       __global       float2 * res) {
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    float2 sum = 0;
    float2 kernelSum = 0;

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            sum += u(x + dx, y + dy) * kern(1 + dx, 1 + dy);
            kernelSum += fabs(kern(1 + dx, 1 + dy));
        }
    }

    res(x, y) = sum / kernelSum;
}


#define du(x, y) du[(y) * W + x]
#define is_maximum(x, y) is_maximum[(y) * W + x]

__constant int2 dxys[8] = {(int2) (1, 0), (int2) (1, 1), (int2) (0, 1), (int2) (-1, 1),
                           (int2) (-1, 0), (int2) (-1, -1), (int2) (0, -1), (int2) (1, -1)};

// Implemented like any non maximum supression (see Canny edge detector)
__kernel void non_maximum_suppression(__global const int    * u,
                                      __global const float2 * du,
                                      __global       int    * is_maximum) {
    int x = (int) get_global_id(0);
    int y = (int) get_global_id(1);

    float anglePi = atan2pi(du(x, y).y, du(x, y).x);
    int direction = (int) round(anglePi / 0.25);
    direction = (direction % 8 + 8) % 8;

    int2 dxy = dxys[direction];
    if (u(x, y) >= u(x + dxy.x, y + dxy.y) && u(x, y) >= u(x - dxy.x, y - dxy.y)) {
        is_maximum(x, y) = 1;
    } else {
        is_maximum(x, y) = 0;
    }
}
