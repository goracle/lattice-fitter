
#include <stdlib.h>     /* strtod */
#include <ostream>
#include <iostream>
#include <gsl/gsl_vector.h>
#include "luscherzeta.h"
#include <math.h>

int main(int argc, char* argv[])
{
  if(argc > 1 && argc!=4 && argc!= 8){
    printf("wrong number of arguments; expected E_pipi, m_pi, L (optional:boost)\n");
    exit(-1);
  }else if(argc == 1){
    ZetaTesting z;
    exit(0);
  }

  double E_pipi = strtod(argv[1], NULL);
  double m_pi = strtod(argv[2], NULL);
  double L_box = strtod(argv[3], NULL);
  double gamma = 1.0;
  GSLvector boost(3);
  if(argc==4){boost[0] = 0; boost[1] = 0; boost[2] = 0;}
  else if(argc == 8) {
    //printf("setting boost argv length = %d\n", argc);
    for(int i=0; i<3; i++){
      boost[i] = strtod(argv[4+i], NULL);
    }
    gamma = strtod(argv[7], NULL);
  }else{
    printf("boost vector input invalid (needs 3 ints)\n");
    exit(-1);
  }

  //printf("calculating phase shift with E_pipi=%e, m_pi=%e, L_box=%e\n", E_pipi, m_pi, L_box);

  double arg = E_pipi*E_pipi/4 - m_pi*m_pi;
  double x = cosh(E_pipi/2)-cosh(m_pi);
  //double arg = acos(1-x);
  //double arg = sqrt(3)*acos(1-x/3);
  //printf("arg=%e, x=%e\n", arg, x);
  //if(arg < 0){printf("E_pipi*E_pipi/4 - m_pi*m_pi<0\n"); exit(-1);}
  bool imag_q = arg < 0;
  if(arg<0) arg = -arg;
  double p_hat = sqrt( arg );
  double p_pipi = 2*asin(p_hat/2);
  //double p_pipi = arg ;
  double q_pipi = L_box * p_pipi /( 2 * M_PI ); //M_PI = pi

  assert(gamma>=1.0);

  //printf("calculating phase shift with p_pipi=%e, q_pipi=%e, test=%e\n", p_pipi, q_pipi, sqrt(E_pipi));

  //printf("boost=%d, %d, %d\n", (int)boost[0], (int)boost[1], (int)boost[2]);

  LuscherZeta zeta;
  //zeta.setMaximumVectorMagnitude(10);
  zeta.setBoost(boost, gamma); //periodic BC's

  double phi = zeta.calcPhi(q_pipi, imag_q)*180/M_PI;
  printf("%.16e", -phi);

  return 0;

  
}
