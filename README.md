# SIMD-and-MT-Accelerated-Fractal-Image-Generation

Using Multi-Threading techniques, and SIMD to accelerate computation time generating fractal images. As well as this, general C++ optimization techniques used to improve performance.

Results: 
Processor: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz (8 Cores, 8 Treads)

-CPU single-thread:	130.731 ms <br />
-CPU multi-thread (8 threads):	31.1229 ms <br />

-SIMD single-thread:	32.9996 ms <br />
-SIMD multi-thread (8 threads):	15.20969 ms <br />

![fractal_SIMD_MT](https://user-images.githubusercontent.com/48512015/128336590-c863618f-49cf-49fb-b69e-11a9b1beb98c.jpg)
(Expected output for images (conveted to JPG from BMP) {Example extracted from output of SIMD-MT))
