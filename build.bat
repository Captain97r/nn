@echo off
rem Build the assembly MLP and its test driver.
rem Requires the MinGW-w64 toolchain (w64devkit). Override with set W64=...
setlocal
if not defined W64 set W64=D:\w64devkit\bin
set W64LIB=%W64%\..\lib
if not exist build mkdir build
"%W64%\x86_64-w64-mingw32-as.exe" src\neural.s -o build\neural.o || goto :err
"%W64%\x86_64-w64-mingw32-gcc.exe" -O2 -Itest test\test.c build\neural.o -o build\test.exe || goto :err
"%W64%\x86_64-w64-mingw32-gcc.exe" -O2 train_mnist.c build\neural.o -o build\train_mnist.exe || goto :err
"%W64%\x86_64-w64-mingw32-gcc.exe" -O2 -shared build\neural.o -o build\neural.dll -Wl,--export-all-symbols -lmsvcrt || goto :err
echo build ok
exit /b 0
:err
echo BUILD FAILED
exit /b 1
