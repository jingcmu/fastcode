#include <iostream>
using namespace std;
int main( )
{
  #pragma omp parallel
  {
    cout << "All threads run this\n";
    #pragma omp sections
    {
      #pragma omp section
      {
        cout << "This executes in parallel\n";
      }
      #pragma omp section
      {
        cout << "Sequential statement 1\n";
        cout << "This always executes after statement 1\n";
      }
      #pragma omp section
      {
        cout << "This also executes in parallel\n";
      }
    }
  }
}