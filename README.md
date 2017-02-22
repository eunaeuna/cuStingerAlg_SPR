# cuStingerAlg_SPR
Streaming PageRank added to cuStingerAlg 

1. cuStinger needs to be installed first
   git clone --recursive https://github.com/cuStinger/cuStinger.git
   mkdir build
   cd build
   ccmake ..
      "CUDA_SDK_ROOT_DIR CUDA_SDK_ROOT_DIR-NOTFOUND" --> "CUDA_SDK_ROOT_DIR  /usr/local/cuda"
   cmake .
   make -j8
   
2. cuStingerAlg_SPR
   git clone --recursive 
   mkdir build
   cd build
   ccmake ..
      "CUDA_SDK_ROOT_DIR CUDA_SDK_ROOT_DIR-NOTFOUND" --> "CUDA_SDK_ROOT_DIR  /usr/local/cuda"
   cmake .
   
   cd ../externals/cuStinger
   mkdir build
   cd build
   ccmake ..
      "CUDA_SDK_ROOT_DIR CUDA_SDK_ROOT_DIR-NOTFOUND" --> "CUDA_SDK_ROOT_DIR  /usr/local/cuda"
   cmake . 
   make -j8
   
   cd ../../../build
   make -j8
   
3. test
   ./cus-alg-tester /data/dimacs/matrix/audikw1.graph
