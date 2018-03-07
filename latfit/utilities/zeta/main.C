
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
  double m_pi = strtod(argv[2], NULL);
  double L_box = strtod(argv[3], NULL);

  //printf("calculating phase shift with E_pipi=%e, m_pi=%e, L_box=%e\n", E_pipi, m_pi, L_box);

  double arg = E_pipi*E_pipi/4 - m_pi*m_pi;
  //if(arg < 0){printf("E_pipi*E_pipi/4 - m_pi*m_pi<0\n"); exit(-1);}
  bool imag_q = arg < 0;
  if(arg<0) arg = -arg;
  double p_pipi = sqrt( arg );
  double q_pipi = L_box * p_pipi /( 2 * M_PI ); //M_PI = pi

  //printf("calculating phase shift with p_pipi=%e, q_pipi=%e, test=%e\n", p_pipi, q_pipi, sqrt(E_pipi));

  LuscherZeta zeta;
  zeta.setTwists(0,0,0); //periodic BC's

  double phi = zeta.calcPhi(q_pipi, imag_q)*180/M_PI;
  printf("%.16e", -phi);

  return 0;

  
}
