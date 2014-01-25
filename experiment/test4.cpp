int main( )
{
	int a[1000000], b[1000000]; 
	// ... some initialization code for populating arrays a and b; 
	int c[1000000];
	#pragma omp parallel for 
	for (int i = 0; i < 1000000; ++i)
	  c[i] = a[i] * b[i] + a[i-1] * b[i+1];
	// ... now do some processing with array c
 }