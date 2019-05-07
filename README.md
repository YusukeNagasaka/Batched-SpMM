Batched Sparse Matrix Multiplication for Graph Convolutional Networks
===

## Versions
0.9 (May, 2019)

## Introduction
This libraries provides high performance Batched Sparse Matrix Multiplication (SpMM) kernel for GPU. Target matrix is small, where the number of row (or column) is tens or hundreds. Such operation can be found in the applications of Graph Convolutional Networks.
The details of Batched SpMM algorithm can be found at the paper (1).

(1) Yusuke Nagasaka, Akira Nukada, Ryosuke Kojima, Satoshi Matsuoka, “Batched Sparse Matrix Multiplication for Accelerating Graph Convolutional Networks”, The 19th Annual IEEE/ACM International Symposium in Cluster, Cloud, and Grid Computing (CCGrid 2019), Larnaca, Cyprus, 2019. (Paper is also at [arXiv](https://arxiv.org/abs/1903.11409))

## Components
batched.cu --- Main kernel code written with CUDA
batched_call.py --- Used for calling batched SpMM kernel in with TensorFlow
Makefile --- Generating object file from batched.cu

## How to use
Detailed instructions will be available soon.
