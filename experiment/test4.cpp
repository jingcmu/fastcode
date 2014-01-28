#include <omp.h>
#include <iostream>
#include <time.h>
using namespace std;

int main( )
{
	int a[100000], b[100000]; 
	// ... some initialization code for populating arrays a and b; 
	int c[100000];
	clock_t start, end;
	start = clock();
	#pragma omp parallel for 
	for (int i = 0; i < 10000000; ++i)
	  c[i%100000] = a[i%100000] * b[i%100000] + a[(i%100000)-1] * b[(i%100000)+1];
	// ... now do some processing with array c
	end = clock();
	cout << end << " " << start << endl;
}