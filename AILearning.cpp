#include <stdio.h>
#include <External/tensorflow/include/c/c_api.h>
#include <windows.h>

#pragma comment(lib, "External/lib/tensorflow.lib")

int main()
{
	printf("Hello from TensorFlow C library version %s\n", TF_Version());

	return 0;
}