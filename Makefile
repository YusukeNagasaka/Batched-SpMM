CC = g++
NVCC = nvcc

CFLAG = -std=c++11 -shared -O3
NFLAG = $(CFLAG) -arch=sm_60 -rdc=true --expt-relaxed-constexpr

batched:
	$(eval TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
	$(eval TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
	$(NVCC) $(NFLAG) batched.cu -o batched.so -Xcompiler -fPIC $(TF_CFLAGS) $(TF_LFLAGS)
