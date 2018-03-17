
#include <stdlib.h>     /* strtod */
#include <ostream>
#include <iostream>
#include "luscherzeta.h"

int main(int argc, char* argv[])
{
  if(argc!=4){
    printf("wrong number of arguments; expected E_pipi, m_pi, L\n");
    exit(-1);
  }

  double E_pipi = strtod(argv[1], NULL);
  double s = E_pipi*E_pipi;
  size_t I = strtol(argv[3], NULL, 10);
  double m_pi = strtod(argv[2], NULL);


  double phi_pheno = PhenoCurve::compute(s, I, m_pi)*180/M_PI;

  printf("%.16e", phi_pheno);

  return 0;

  
}
