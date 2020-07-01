# KFilter - Free C++ Extended Kalman Filter Library 

This code is a modified version of the source code found at
http://kalman.sourceforge.net/

It has been converted to use CMake as the build system and with updated
comments.

For installation instructions, read INSTALL.txt.

To use the kalman classes :

```
#include "kalman/ekfilter.hpp"

```

Don't forget to link the .a (-lkalman) on UNIX/Linux or the .lib on Windows. 
The include path must be accessible (nothing to do on UNIX/Linux, set the 
include path on Windows).

HTML documentation (with a code example and procedure) is located in cmake
build subdirectory doc/public/html/index.html.

Samples combining matlab files and C++ are available in the 'examples' 
directory.

For further information, go to the SourceForge project website :

https://github.com/sfmontyo/KalmanFilter



