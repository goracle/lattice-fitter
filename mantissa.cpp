//partial sums of the harmonic series
//computed up and down to show differing precisions
//Summer Study group, 2015

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

static void show_usage(std::string name)
{
  std::cerr << "Usage: " << name << " <N>" << "\n"
	    << "See comments, partial sum of harmonic series"
	    << "\n";
}

int main(int argc, char* argv[])
{
  if (argc != 2)
    {
      show_usage(argv[0]);
      return 2;
    }
  double sumdown=0;
  double sumup=0;
  char *endptr;
  long int N = strtol(argv[1], &endptr, 10);
  if (!*argv[1] || *endptr)
    {
      cerr << "Invalid number " << argv[1] << '\n';
      return 2;
    }
  for(int i=1; i<N+1; i++)
    {
      sumup+=1.0/i;
    }
  for(int i=N; i>0; i--)
    {
      sumdown+=1.0/i;
    }
  cout << "sum down(" << N << ") = " << sumdown << "\n";
  cout << "sum up (" << N << ") = " << sumup << "\n";
  return 0;
}
	 
