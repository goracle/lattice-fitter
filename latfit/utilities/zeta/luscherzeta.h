#ifndef LUSCHER_ZETA_H
#define LUSCHER_ZETA_H
#include <gsl/gsl_integration.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>
#include <assert.h>

//C.Kelly 2014
//Classes and methods to determine and manipulate Luscher's \zeta and \phi functions for use in the quantization condition and Lellouch-Luscher factor
//This calculaton method is based on Qi Liu's Matlab code which in turn was based on Physical review D 70 074513

//Must link against gsl and gslcblas libraries

//A wrapper class around GSL vectors. Note, this does not check vector dimensions during operations, so be careful!
class GSLvector{
  gsl_vector * v;
  
public:
  inline explicit GSLvector(const int d): v(gsl_vector_calloc(d)){  } //Initialized to zero
  inline explicit GSLvector(const int d, const double init): v(gsl_vector_alloc(d)){ gsl_vector_set_all(v,init); }

  inline const int dim() const{ return v->size; }

  inline GSLvector(const GSLvector &r){ 
    v = gsl_vector_alloc(r.dim());
    gsl_vector_memcpy (v,r.v);
  }

  inline const double & operator[](const int i) const{ return *gsl_vector_const_ptr(v,i); }
  inline double & operator[](const int i) { return *gsl_vector_ptr(v,i); }

  inline GSLvector & operator+=(const GSLvector &r){
    gsl_vector_add(v,r.v); return *this;
  }
  inline GSLvector & operator-=(const GSLvector &r){
    gsl_vector_sub(v,r.v); return *this;
  }
  
  //a[i] = a[i]*b[i]
  inline GSLvector & outer_prod(const GSLvector &r){
    gsl_vector_mul(v,r.v); return *this;
  }

  //a[i] = a[i]/b[i]
  inline GSLvector & outer_div(const GSLvector &r){
    gsl_vector_div(v,r.v); return *this;
  }  

  inline GSLvector & operator*=(const double x){
    gsl_vector_scale(v,x); return *this;
  }

  //a[i] = a[i] + x
  inline GSLvector & operator+=(const double x){
    gsl_vector_add_constant(v,x); return *this;
  }

  //Compute the Euclidean norm ||x||_2 = \sqrt {\sum x_i^2} of the vector x. 
  inline double norm() const{ 
    return gsl_blas_dnrm2 (v);
  }
  //Squared norm
  inline double norm2() const{
    double nrm = norm();
    return nrm*nrm;
  }

  inline ~GSLvector(){
    gsl_vector_free(v);
  }

  friend inline double dot(const GSLvector &a, const GSLvector &b);
};
inline std::ostream & operator<<(std::ostream &os, const GSLvector &v){
  os << "(" << v[0] << "," << v[1] << "," << v[2] << ")"; return os;
}


inline double dot(const GSLvector &a, const GSLvector &b){
  double out;
  gsl_blas_ddot (a.v,b.v,&out);
  return out;
}

inline GSLvector operator-(const GSLvector &a, const GSLvector &b){
  GSLvector out(a); out -= b; return out;
}

inline GSLvector operator*(const double x, const GSLvector &v){
  GSLvector out(v); out *= x; return out;
}
inline GSLvector operator*(const GSLvector &v,const double x){
  GSLvector out(v); out *= x; return out;
}



inline double square(const double x){ return x*x; }

//Luscher's quantization condition is
//n*\pi - \delta_0(k) = \phi(q) 
//where q = kL/2\pi  and  k is the solution of E_{\pi\pi} = 2\sqrt( m_\pi^2 + k^2 )

//Here  \tan(\phi(q)) = -\pi^{3/2}q / \zeta_{00}(1;q^2)
//This class computed \zeta_{00} and by extension, \phi, as a function of q^2, for periodic/H-parity/G-parity BCs in arbitrary spatial directions

class LuscherZeta{
  //Private variables
  GSLvector d; //H-parity or G-parity twist directions (x,y,z): there is a twist, use 1.0 or else use 0.0 
  GSLvector dnorm; //Normalised BC vector
  int N; //First integral is over integer-component vectors in Z^3. The input parameter 'N" sets the maximum magnitude of these vectors

  //Errors on the integrals
  double epsabs;
  double epsrel;

  //Private methods
  struct zeta_params{
    double q2;
    double gn2;
  };
  static double zeta_func(double t, void *zparams){
    const static double pi2 = M_PI*M_PI;
    zeta_params *p = (zeta_params*)zparams;
    return exp( t*p->q2 - pi2 * p->gn2/t ) * pow(M_PI/t, 1.5);
  };

  //If non-null, abserr is the absolute error on the result, and nevals is the number of integration steps used
  double int_zeta(const double q2, const double gn2, double *abserr = NULL, size_t *nevals = NULL) const{
    //Prepare the zeta function
    zeta_params zeta_p;
    zeta_p.q2 = q2;
    zeta_p.gn2 = gn2;

    gsl_function zeta;
    zeta.function = &zeta_func;
    zeta.params = &zeta_p;

    //do the integral
    gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc(100);
    
    double start = 1e-06;
    double end = 1.0;

    double result;
    gsl_integration_cquad(&zeta, start, end, epsabs, epsrel, workspace, &result, abserr, nevals);
    gsl_integration_cquad_workspace_free(workspace);
    return result;
  }

  double z3sum(const double q, const GSLvector &n, const bool imag_q = false) const{
    double r2 = square(n[0]+d[0]/2.0)+square(n[1]+d[1]/2.0)+square(n[2]+d[2]/2.0);
    double q2 = imag_q ? -square(q) : square(q);
    double out = 0.0;
        
    //printf("z3sum with q = %f and n = %f %f %f\n",q,n[0],n[1],n[2]);

    //sum over e^{-(r^2 - q^2)}/(r^2 - q^2)        
    out += exp(q2-r2) / (r2-q2);

    //skip integral for 0 vector
    if(n[0]==0.0 && n[1]==0.0 && n[2]==0.0)
      return out;

    //integral \int_0^1 dt e^{q^2t} (\pi/t)^{3/2} e^{-\pi^2 r^2/t} modified for APBC where appropriate (modifies r (aka n) )
    //define n_perp = n - (n \cdot dnorm) dnorm  is perpendicular to d
    double dcoeff = dot(n,dnorm);
    GSLvector np = n - dcoeff*dnorm;
    double gn2 = square(dcoeff) + np.norm2();

    double int_n = int_zeta(q*q,gn2);
    out += int_n*pow(-1, dot(n,d));

    return out;
  }
  
 public:
  LuscherZeta(): d(3), dnorm(3), N(5), epsabs(1e-06),epsrel(1e-06) {}
  LuscherZeta(const double x, const double y, const double z): d(3), dnorm(3), N(5), epsabs(1e-06),epsrel(1e-06){
    setTwists(x,y,z);
  }

  void setTwists(const double x, const double y, const double z){
    d[0] = x; d[1] = y; d[2] = z;
    double dnrm = d.norm();
    if(d[0]==d[1]==d[2]==0) dnrm = 1;
    
    for(int i=0;i<3;i++){
      if(d[i] != 0.0 && d[i] != 1.0){ std::cout << "LuscherZeta::setTwists : Error, arguments must be 0 or 1\n"; exit(-1); }
      dnorm[i] = double(d[i])/dnrm; //normalize d
    }
  }

  inline void setIntegrationErrorBounds(const double eps_abs, const double eps_rel){
    epsabs = eps_abs; epsrel = eps_rel;
  }
     
  inline void setMaximumVectorMagnitude(const int iN){
    N = iN;
  }
        
  double calcZeta00(const double q, const bool imag_q = false) const{
    //q is a scalar modulus
    double result = 0.0;
    //outer loop over vectors in Z^3
    for(int nx = -N; nx <= N; ++nx){
      int Ny( floor(sqrt( double(N*N - nx*nx) )) );
      for(int ny = -Ny; ny <= Ny; ++ny){
	int Nz( floor(sqrt( double(N*N - nx*nx - ny*ny) ) + 0.5 ) ); //round to nearest int
	for(int nz = -Nz; nz <= Nz; ++nz){
	  GSLvector n(3);
	  n[0] = nx; n[1] = ny; n[2] = nz;
	  result += z3sum(q,n, imag_q);
	}
      }
    }
    
    //constant part
    double const_part = 0.0;
    double q2 = square(q);
    bool warn = true;
    for(unsigned int l=0;l<100;l++){
      double c_aux = pow(q2,l) / gsl_sf_fact(l) / (l-0.5);
      if(l>9 && c_aux < 1e-08 * fabs(const_part)){
	warn = false;
	break;
      }
      const_part += c_aux;
    }
    if(warn) printf("Warning: reaches the maximum loop number when doing the summation\n");

    result += pow(M_PI,1.5)*const_part;
    result /= sqrt(4*M_PI);
    return result;
  }

  inline double calcPhi(const double q, const bool imag_q = false) const{
    return atan(-q*pow(M_PI,1.5)/calcZeta00(q, imag_q));
  }

  inline double calcPhiDeriv(const double q, const double frac_shift = 1e-04) const{
    double dq = frac_shift * q;
    return ( calcPhi(q+dq) - calcPhi(q-dq) )/(2.*dq);
  }
};

//Test by comparing with numbers computed by Daiqian (based off Qi's MatLab code) 
class ZetaTesting{
public:
  ZetaTesting(){
    LuscherZeta zeta;
    
    //q  dx dy dz phi
    //0.100000	0	0	1	0.790456
    zeta.setTwists(0,0,1);
    printf("0.100000	0	0	1	Expect 0.790456 Got %.6f\n", zeta.calcPhi(0.1));
      
    //0.300000	1	1	1	1.079671
    zeta.setTwists(1,1,1);
    printf("0.300000	1	1	1	Expect 1.079671 Got %.6f\n", zeta.calcPhi(0.3));

    //0.400000	0	0	0	0.571791
    zeta.setTwists(0,0,0);
    printf("0.400000	0	0	0	Expect 0.571791  Got %.6f\n", zeta.calcPhi(0.4));

    //2.000000	1	1	1	-1.315088
    zeta.setTwists(1,1,1);
    printf("2.000000	1	1	1	Expect -1.315088 Got %.6f\n", zeta.calcPhi(2.0));
  }
};

class PhenoCurve{
public:
  //Calculates \delta_0 pi-pi scattering length in radians
  //s is the Mandelstam variable s=E_pipi^2  *in MeV^2*
  //I is the isospin : 0 or 2
  //mpi in MeV
  //fitcurve is A, B or C - these are the three curves in Schenk, Nucl.Phys. B363 (1991) 97-116
  //                        for I=0 the curves are upper, best and lower bounds respectively, and for I=2 they are lower, best and upper bounds respectively
  static double compute(const double &s, const int &I, const double &mpi, const char fitcurve = 'B'){
    double _4mpi2 = 4*pow(mpi,2);
    double sbrack = (s - _4mpi2)/_4mpi2;
    double gammarho = 149;
    double mrho = 769;
    double mrho2 = pow(769, 2);
    //printf("(s-4m_pi^2)/(4m_pi^2) = %f\n",sbrack);
    assert(I==0 || I==1 || I==2);
    if(I==1) assert(sqrt(s)<769); //only valid below inelastic threshold
    
    int i= (I == 0 ? 0 : 1);
    int fci;
    if(fitcurve == 'A') fci = 0;
    else if(fitcurve == 'B') fci = 1;
    else if(fitcurve == 'C') fci = 2;    
    else{ printf("PhenoCurve::compute invalid fitcurve\n"); exit(-1); }

    //Coefficients for I=0,2 respectively
    double a[2] = {0.2, -0.042};
    double b[2] = {0.24, -0.075};
    double sl[3][2] = {
      {pow(840,2), -pow(1000,2)},
      {pow(865,2), -pow(790,2)}, 
      {pow(890,2), -pow(680,2)}
    }; //A,B,C curves resp, units are MeV
    double c[3][2] = {
      {0.008,0.0},
      {0.0,0.0},
      {-0.015,0.0} 
    };

    double tandelta;
    if(I == 0 || I == 2)
      tandelta = sqrt(1. - _4mpi2/s) * ( a[i] + b[i]*sbrack + c[fci][i]*pow(sbrack,2) ) * (_4mpi2 - sl[fci][i])/(s-sl[fci][i]);
    else //I=1
      tandelta = ((_4mpi2-mrho2)/(s-mrho2))*(mrho*gammarho/(mrho2-_4mpi2));
    return atan(tandelta);
  }
  static double compute_deriv(const double &s, const int &I, const double &mpi, const char fitcurve = 'B', const double &frac_shift = 1e-04){
    double ds = frac_shift * s;
    return ( compute(s+ds,I,mpi,fitcurve) - compute(s-ds,I,mpi,fitcurve) )/(2.*ds);
  }

};

#endif
