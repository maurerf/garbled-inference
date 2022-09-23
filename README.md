[![License](https://img.shields.io/badge/license-Unlicense-blue.svg)](https://github.com/maurerf/garbled-inference/blob/master/LICENSE)
 


# Garbled Inference: Modular privacy-preserving activation functions
Secure multi-party computational approach to activation layers of pre-trained deep learning classifiers using Yao's Garbled Circuit Protocol [1].
The implementation of that protocol used here is a modified version of TinyGarble [2].

# Build and Usage
Compile the two binaries.

``` bash
$ mkdir build && cd build
$ cmake ..
$ make client
$ make server 
```

Now in two separate concurrent processes, start the client application and then the server.
``` bash
$ ./client
```
``` bash
$ ./server
```

# Documentation
Documentation of this project can be found [here](example.com).

# Project template
This project was built upon the [cpp-project](https://github.com/bsamseth/cpp-project) GitHub repository template.

All content added in this repositories initial commit is provided by them. Their project is publically available under the Unlicensed license.

# References
[1] YAO, Andrew Chi-Chih: How to generate and exchange secrets. In: 27th
Annual Symposium on Foundations of Computer Science (sfcs 1986) IEEE, 1986,
S. 162–167

[2] https://github.com/esonghori/TinyGarble
