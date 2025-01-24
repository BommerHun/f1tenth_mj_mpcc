/* This file was automatically generated by CasADi 3.6.7.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) f1tenth_MJPC_cost_ext_cost_fun_jac_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_c2 CASADI_PREFIX(c2)
#define casadi_c3 CASADI_PREFIX(c3)
#define casadi_c4 CASADI_PREFIX(c4)
#define casadi_c5 CASADI_PREFIX(c5)
#define casadi_c6 CASADI_PREFIX(c6)
#define casadi_c7 CASADI_PREFIX(c7)
#define casadi_c8 CASADI_PREFIX(c8)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_clear_casadi_int CASADI_PREFIX(clear_casadi_int)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_de_boor CASADI_PREFIX(de_boor)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_f3 CASADI_PREFIX(f3)
#define casadi_f4 CASADI_PREFIX(f4)
#define casadi_f5 CASADI_PREFIX(f5)
#define casadi_f6 CASADI_PREFIX(f6)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_fill_casadi_int CASADI_PREFIX(fill_casadi_int)
#define casadi_low CASADI_PREFIX(low)
#define casadi_nd_boor_eval CASADI_PREFIX(nd_boor_eval)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s11 CASADI_PREFIX(s11)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_de_boor(casadi_real x, const casadi_real* knots, casadi_int n_knots, casadi_int degree, casadi_real* boor) {
  casadi_int d, i;
  for (d=1;d<degree+1;++d) {
    for (i=0;i<n_knots-d-1;++i) {
      casadi_real b, bottom;
      b = 0;
      bottom = knots[i + d] - knots[i];
      if (bottom) b = (x - knots[i]) * boor[i] / bottom;
      bottom = knots[i + d + 1] - knots[i + 1];
      if (bottom) b += (knots[i + d + 1] - x) * boor[i + 1] / bottom;
      boor[i] = b;
    }
  }
}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

void casadi_fill_casadi_int(casadi_int* x, casadi_int n, casadi_int alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_clear_casadi_int(casadi_int* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

casadi_int casadi_low(casadi_real x, const casadi_real* grid, casadi_int ng, casadi_int lookup_mode) {
  switch (lookup_mode) {
    case 1:
      {
        casadi_real g0, dg;
        casadi_int ret;
        g0 = grid[0];
        dg = grid[ng-1]-g0;
        ret = (casadi_int) ((x-g0)*(ng-1)/dg);
        if (ret<0) ret=0;
        if (ret>ng-2) ret=ng-2;
        return ret;
      }
    case 2:
      {
        casadi_int start, stop, pivot;
        if (ng<2 || x<grid[1]) return 0;
        if (x>grid[ng-1]) return ng-2;
        start = 0;
        stop  = ng-1;
        while (1) {
          pivot = (stop+start)/2;
          if (x < grid[pivot]) {
            if (pivot==stop) return pivot;
            stop = pivot;
          } else {
            if (pivot==start) return pivot;
            start = pivot;
          }
        }
      }
    default:
      {
        casadi_int i;
        for (i=0; i<ng-2; ++i) {
          if (x < grid[i+1]) break;
        }
        return i;
      }
  }
}

void casadi_nd_boor_eval(casadi_real* ret, casadi_int n_dims, const casadi_real* all_knots, const casadi_int* offset, const casadi_int* all_degree, const casadi_int* strides, const casadi_real* c, casadi_int m, const casadi_real* all_x, const casadi_int* lookup_mode, casadi_int* iw, casadi_real* w) {
  casadi_int n_iter, k, i, pivot;
  casadi_int *boor_offset, *starts, *index, *coeff_offset;
  casadi_real *cumprod, *all_boor;
  boor_offset = iw; iw+=n_dims+1;
  starts = iw; iw+=n_dims;
  index = iw; iw+=n_dims;
  coeff_offset = iw;
  cumprod = w; w+= n_dims+1;
  all_boor = w;
  boor_offset[0] = 0;
  cumprod[n_dims] = 1;
  coeff_offset[n_dims] = 0;
  n_iter = 1;
  for (k=0;k<n_dims;++k) {
    casadi_real *boor;
    const casadi_real* knots;
    casadi_real x;
    casadi_int degree, n_knots, n_b, L, start;
    boor = all_boor+boor_offset[k];
    degree = all_degree[k];
    knots = all_knots + offset[k];
    n_knots = offset[k+1]-offset[k];
    n_b = n_knots-degree-1;
    x = all_x[k];
    L = casadi_low(x, knots+degree, n_knots-2*degree, lookup_mode[k]);
    start = L;
    if (start>n_b-degree-1) start = n_b-degree-1;
    starts[k] = start;
    casadi_clear(boor, 2*degree+1);
    if (x>=knots[0] && x<=knots[n_knots-1]) {
      if (x==knots[1]) {
        casadi_fill(boor, degree+1, 1.0);
      } else if (x==knots[n_knots-1]) {
        boor[degree] = 1;
      } else if (knots[L+degree]==x) {
        boor[degree-1] = 1;
      } else {
        boor[degree] = 1;
      }
    }
    casadi_de_boor(x, knots+start, 2*degree+2, degree, boor);
    boor+= degree+1;
    n_iter*= degree+1;
    boor_offset[k+1] = boor_offset[k] + degree+1;
  }
  casadi_clear_casadi_int(index, n_dims);
  for (pivot=n_dims-1;pivot>=0;--pivot) {
    cumprod[pivot] = (*(all_boor+boor_offset[pivot]))*cumprod[pivot+1];
    coeff_offset[pivot] = starts[pivot]*strides[pivot]+coeff_offset[pivot+1];
  }
  for (k=0;k<n_iter;++k) {
    casadi_int pivot = 0;
    for (i=0;i<m;++i) ret[i] += c[coeff_offset[0]+i]*cumprod[0];
    index[0]++;
    {
      while (index[pivot]==boor_offset[pivot+1]-boor_offset[pivot]) {
        index[pivot] = 0;
        if (pivot==n_dims-1) break;
        index[++pivot]++;
      }
      while (pivot>0) {
        cumprod[pivot] = (*(all_boor+boor_offset[pivot]+index[pivot]))*cumprod[pivot+1];
        coeff_offset[pivot] = (starts[pivot]+index[pivot])*strides[pivot]+coeff_offset[pivot+1];
        pivot--;
      }
    }
    cumprod[0] = (*(all_boor+index[0]))*cumprod[1];
    coeff_offset[0] = (starts[0]+index[0])*m+coeff_offset[1];
  }
}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[2] = {0, 28};
static const casadi_int casadi_s1[1] = {2};
static const casadi_int casadi_s2[1] = {1};
static const casadi_int casadi_s3[1] = {0};
static const casadi_int casadi_s4[2] = {0, 30};
static const casadi_int casadi_s5[1] = {3};
static const casadi_int casadi_s6[2] = {0, 26};
static const casadi_int casadi_s7[30] = {26, 1, 0, 26, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
static const casadi_int casadi_s8[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s9[4] = {0, 1, 0, 0};
static const casadi_int casadi_s10[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s11[37] = {33, 1, 0, 33, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

static const casadi_real casadi_c0[28] = {0., 0., 0., 2.0667941717248488e+00, 3.1001912575872730e+00, 4.1335883434496976e+00, 5.1669854293121222e+00, 6.2003825151745460e+00, 7.2337796010369706e+00, 8.2671766868993952e+00, 9.3005737727618190e+00, 1.0333970858624244e+01, 1.1367367944486668e+01, 1.2400765030349092e+01, 1.3434162116211517e+01, 1.4467559202073941e+01, 1.5500956287936367e+01, 1.6534353373798790e+01, 1.7567750459661216e+01, 1.8601147545523638e+01, 1.9634544631386063e+01, 2.0667941717248489e+01, 2.1701338803110911e+01, 2.2734735888973336e+01, 2.3768132974835762e+01, 2.5834927146560609e+01, 2.5834927146560609e+01, 2.5834927146560609e+01};
static const casadi_real casadi_c1[25] = {6.5443756480842052e-01, 9.2943035689597087e-01, 6.9510141057043140e-01, 7.7975352210089666e-01, 9.1053199132018570e-01, 4.0471754811644356e-01, -1.1557994616646283e-01, -7.2228738328437236e-01, -9.1527221933827718e-01, -6.5030371297517964e-01, -8.2692262294871433e-01, -7.9412905853161941e-01, -7.2672155451214682e-01, -7.9412905853161542e-01, -8.2692262294872054e-01, -6.5030371297517164e-01, -9.1527221933828429e-01, -7.2228738328437503e-01, -1.1557994616645040e-01, 4.0471754811644534e-01, 9.1053199132018081e-01, 7.7975352210089932e-01, 6.9510141057042851e-01, 9.2943035689597120e-01, 6.5443756480843951e-01};
static const casadi_real casadi_c2[25] = {4.1663687796189852e-01, 1.2665917233411434e+00, 5.0688521992966962e-01, -3.2779268158395936e-01, -9.5080199516552522e-01, -7.7730276165019518e-01, -6.8266427365053373e-01, -9.1105947854467284e-01, -7.5534817312963543e-01, 1.2777970274869821e-01, 8.7596020136519637e-01, 8.4114041055290056e-01, 6.7965523173592857e-01, 8.4114041055289746e-01, 8.7596020136519637e-01, 1.2777970274870309e-01, -7.5534817312963209e-01, -9.1105947854467706e-01, -6.8266427365053495e-01, -7.7730276165019041e-01, -9.5080199516552744e-01, -3.2779268158396180e-01, 5.0688521992966851e-01, 1.2665917233411457e+00, 4.1663687796190807e-01};
static const casadi_real casadi_c3[30] = {0., 0., 0., 0., 2.0667941717248488e+00, 3.1001912575872730e+00, 4.1335883434496976e+00, 5.1669854293121222e+00, 6.2003825151745460e+00, 7.2337796010369706e+00, 8.2671766868993952e+00, 9.3005737727618190e+00, 1.0333970858624244e+01, 1.1367367944486668e+01, 1.2400765030349092e+01, 1.3434162116211517e+01, 1.4467559202073941e+01, 1.5500956287936367e+01, 1.6534353373798790e+01, 1.7567750459661216e+01, 1.8601147545523638e+01, 1.9634544631386063e+01, 2.0667941717248489e+01, 2.1701338803110911e+01, 2.2734735888973336e+01, 2.3768132974835762e+01, 2.5834927146560609e+01, 2.5834927146560609e+01, 2.5834927146560609e+01, 2.5834927146560609e+01};
static const casadi_real casadi_c4[26] = {6.6996477657936486e-05, 2.8710122017675427e-01, 1.5959934160549578e+00, 2.2944116949110307e+00, 1.9556716929951374e+00, 9.7311568195890474e-01, 1.6985327323677846e-01, -5.3560999777607166e-01, -1.4770962079514764e+00, -2.2576708088751478e+00, -2.1256236364222763e+00, -1.2204089170000199e+00, -3.5117686793352915e-01, 3.5117686793353009e-01, 1.2204089170000181e+00, 2.1256236364222745e+00, 2.2576708088751509e+00, 1.4770962079514836e+00, 5.3560999777607443e-01, -1.6985327323677701e-01, -9.7311568195889842e-01, -1.9556716929951332e+00, -2.2944116949110289e+00, -1.5959934160549576e+00, -2.8710122017675171e-01, -6.6996477649022989e-05};
static const casadi_real casadi_c5[26] = {3.3968943834347052e-04, 4.5120227100629234e-01, 1.4116728933346616e+00, 2.3694272560844540e+00, 3.1752222735144824e+00, 4.1161633799292723e+00, 4.5343973147501906e+00, 4.4149573351976326e+00, 3.6685476581563661e+00, 2.7227080139213569e+00, 2.0506860520072920e+00, 1.1961466232183784e+00, 3.7549596833313215e-01, -3.7549596833313126e-01, -1.1961466232183737e+00, -2.0506860520072938e+00, -2.7227080139213511e+00, -3.6685476581563670e+00, -4.4149573351976361e+00, -4.5343973147501817e+00, -4.1161633799292616e+00, -3.1752222735144766e+00, -2.3694272560844456e+00, -1.4116728933346570e+00, -4.5120227100628735e-01, -3.3968943832575596e-04};
static const casadi_real casadi_c6[26] = {0., 0., 2.0667941717248488e+00, 3.1001912575872730e+00, 4.1335883434496976e+00, 5.1669854293121222e+00, 6.2003825151745460e+00, 7.2337796010369706e+00, 8.2671766868993952e+00, 9.3005737727618190e+00, 1.0333970858624244e+01, 1.1367367944486668e+01, 1.2400765030349092e+01, 1.3434162116211517e+01, 1.4467559202073941e+01, 1.5500956287936367e+01, 1.6534353373798790e+01, 1.7567750459661216e+01, 1.8601147545523638e+01, 1.9634544631386063e+01, 2.0667941717248489e+01, 2.1701338803110911e+01, 2.2734735888973336e+01, 2.3768132974835762e+01, 2.5834927146560609e+01, 2.5834927146560609e+01};
static const casadi_real casadi_c7[24] = {8.2248620303579889e-01, -4.9010299061530566e-01, -8.0770297587693141e-01, -6.0287504397366432e-01, 1.6789212577519108e-01, 9.1579983429777689e-02, -2.2101398196176558e-01, 1.5067906378416784e-01, 8.5458715527663476e-01, 7.2400097586118306e-01, -3.3694492938536658e-02, -1.5626633849291738e-01, 1.5626633849291438e-01, 3.3694492938539655e-02, -7.2400097586117829e-01, -8.5458715527663620e-01, -1.5067906378417517e-01, 2.2101398196176869e-01, -9.1579983429771805e-02, -1.6789212577519785e-01, 6.0287504397366465e-01, 8.0770297587693185e-01, 4.9010299061530782e-01, -8.2248620303579223e-01};
static const casadi_real casadi_c8[24] = {2.6610563920649566e-01, -1.5117063874820824e-01, 8.1916344344844516e-02, 1.2655200117014798e-01, -4.8946765006755694e-01, -5.0348264128177878e-01, -5.8710000774927673e-01, -1.8674799715817747e-01, 2.5640531600877048e-01, -1.7091098125764193e-01, 3.1733749655125987e-02, 6.5229044035108208e-02, -6.5229044035104322e-02, -3.1733749655135868e-02, 1.7091098125765569e-01, -2.5640531600878513e-01, 1.8674799715818180e-01, 5.8710000774929194e-01, 5.0348264128176778e-01, 4.8946765006755072e-01, -1.2655200117014076e-01, -8.1916344344849845e-02, 1.5117063874821030e-01, -2.6610563920647778e-01};

/* jac_traj:(x,out_f[1x1,0nz])->(jac_f_x) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, w1;
  /* #0: @0 = input[0][0] */
  w0 = arg[0] ? arg[0][0] : 0;
  /* #1: @1 = BSpline(@0) */
  casadi_clear((&w1), 1);
  CASADI_PREFIX(nd_boor_eval)((&w1),1,casadi_c0,casadi_s0,casadi_s1,casadi_s2,casadi_c1,1,(&w0),casadi_s3, iw, w);
  /* #2: output[0][0] = @1 */
  if (res[0]) res[0][0] = w1;
  return 0;
}

/* jac_traj:(x,out_f[1x1,0nz])->(jac_f_x) */
static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, w1;
  /* #0: @0 = input[0][0] */
  w0 = arg[0] ? arg[0][0] : 0;
  /* #1: @1 = BSpline(@0) */
  casadi_clear((&w1), 1);
  CASADI_PREFIX(nd_boor_eval)((&w1),1,casadi_c0,casadi_s0,casadi_s1,casadi_s2,casadi_c2,1,(&w0),casadi_s3, iw, w);
  /* #2: output[0][0] = @1 */
  if (res[0]) res[0][0] = w1;
  return 0;
}

/* traj:(x)->(f) */
static int casadi_f3(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, w1;
  /* #0: @0 = input[0][0] */
  w0 = arg[0] ? arg[0][0] : 0;
  /* #1: @1 = BSpline(@0) */
  casadi_clear((&w1), 1);
  CASADI_PREFIX(nd_boor_eval)((&w1),1,casadi_c3,casadi_s4,casadi_s5,casadi_s2,casadi_c4,1,(&w0),casadi_s3, iw, w);
  /* #2: output[0][0] = @1 */
  if (res[0]) res[0][0] = w1;
  return 0;
}

/* traj:(x)->(f) */
static int casadi_f4(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, w1;
  /* #0: @0 = input[0][0] */
  w0 = arg[0] ? arg[0][0] : 0;
  /* #1: @1 = BSpline(@0) */
  casadi_clear((&w1), 1);
  CASADI_PREFIX(nd_boor_eval)((&w1),1,casadi_c3,casadi_s4,casadi_s5,casadi_s2,casadi_c5,1,(&w0),casadi_s3, iw, w);
  /* #2: output[0][0] = @1 */
  if (res[0]) res[0][0] = w1;
  return 0;
}

/* jac_jac_traj:(x,out_f[1x1,0nz],out_jac_f_x[1x1,0nz])->(jac_jac_f_x_x,jac_jac_f_x_out_f[1x1,0nz]) */
static int casadi_f5(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real *rr, *ss;
  casadi_real w0, w1, w2, w3;
  /* #0: @0 = zeros(2x1,1nz) */
  w0 = 0.;
  /* #1: @1 = input[0][0] */
  w1 = arg[0] ? arg[0][0] : 0;
  /* #2: @2 = BSpline(@1) */
  casadi_clear((&w2), 1);
  CASADI_PREFIX(nd_boor_eval)((&w2),1,casadi_c6,casadi_s6,casadi_s2,casadi_s2,casadi_c7,1,(&w1),casadi_s3, iw, w);
  /* #3: @1 = ones(2x1,1nz) */
  w1 = 1.;
  /* #4: {@3, NULL} = vertsplit(@1) */
  w3 = w1;
  /* #5: @2 = (@2*@3) */
  w2 *= w3;
  /* #6: (@0[0] = @2) */
  for (rr=(&w0)+0, ss=(&w2); rr!=(&w0)+1; rr+=1) *rr = *ss++;
  /* #7: @0 = @0' */
  /* #8: {@2, NULL} = horzsplit(@0) */
  w2 = w0;
  /* #9: output[0][0] = @2 */
  if (res[0]) res[0][0] = w2;
  return 0;
}

/* jac_jac_traj:(x,out_f[1x1,0nz],out_jac_f_x[1x1,0nz])->(jac_jac_f_x_x,jac_jac_f_x_out_f[1x1,0nz]) */
static int casadi_f6(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real *rr, *ss;
  casadi_real w0, w1, w2, w3;
  /* #0: @0 = zeros(2x1,1nz) */
  w0 = 0.;
  /* #1: @1 = input[0][0] */
  w1 = arg[0] ? arg[0][0] : 0;
  /* #2: @2 = BSpline(@1) */
  casadi_clear((&w2), 1);
  CASADI_PREFIX(nd_boor_eval)((&w2),1,casadi_c6,casadi_s6,casadi_s2,casadi_s2,casadi_c8,1,(&w1),casadi_s3, iw, w);
  /* #3: @1 = ones(2x1,1nz) */
  w1 = 1.;
  /* #4: {@3, NULL} = vertsplit(@1) */
  w3 = w1;
  /* #5: @2 = (@2*@3) */
  w2 *= w3;
  /* #6: (@0[0] = @2) */
  for (rr=(&w0)+0, ss=(&w2); rr!=(&w0)+1; rr+=1) *rr = *ss++;
  /* #7: @0 = @0' */
  /* #8: {@2, NULL} = horzsplit(@0) */
  w2 = w0;
  /* #9: output[0][0] = @2 */
  if (res[0]) res[0][0] = w2;
  return 0;
}

/* f1tenth_MJPC_cost_ext_cost_fun_jac:(i0[26],i1[7],i2[0],i3[0])->(o0,o1[33]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real **res1=res+2, *rr, *ss;
  const casadi_real **arg1=arg+4, *cs;
  casadi_real w0, *w1=w+13, w2, w4, w5, w6, *w7=w+43, w8, *w9=w+46, *w10=w+48, w11, w12, *w13=w+52, w14, *w15=w+60;
  /* #0: @0 = 50 */
  w0 = 50.;
  /* #1: @1 = input[0][0] */
  casadi_copy(arg[0], 26, w1);
  /* #2: @2 = @1[12] */
  for (rr=(&w2), ss=w1+12; ss!=w1+13; ss+=1) *rr++ = *ss;
  /* #3: @3 = 00 */
  /* #4: @4 = jac_traj(@2, @3) */
  arg1[0]=(&w2);
  arg1[1]=0;
  res1[0]=(&w4);
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #5: @5 = jac_traj(@2, @3) */
  arg1[0]=(&w2);
  arg1[1]=0;
  res1[0]=(&w5);
  if (casadi_f2(arg1, res1, iw, w, 0)) return 1;
  /* #6: @6 = (-@5) */
  w6 = (- w5 );
  /* #7: @7 = horzcat(@4, @6) */
  rr=w7;
  *rr++ = w4;
  *rr++ = w6;
  /* #8: @7 = @7' */
  /* #9: @6 = traj(@2) */
  arg1[0]=(&w2);
  res1[0]=(&w6);
  if (casadi_f3(arg1, res1, iw, w, 0)) return 1;
  /* #10: @8 = traj(@2) */
  arg1[0]=(&w2);
  res1[0]=(&w8);
  if (casadi_f4(arg1, res1, iw, w, 0)) return 1;
  /* #11: @9 = horzcat(@6, @8) */
  rr=w9;
  *rr++ = w6;
  *rr++ = w8;
  /* #12: @9 = @9' */
  /* #13: @6 = @1[0] */
  for (rr=(&w6), ss=w1+0; ss!=w1+1; ss+=1) *rr++ = *ss;
  /* #14: @8 = @1[1] */
  for (rr=(&w8), ss=w1+1; ss!=w1+2; ss+=1) *rr++ = *ss;
  /* #15: @10 = vertcat(@6, @8) */
  rr=w10;
  *rr++ = w6;
  *rr++ = w8;
  /* #16: @9 = (@9-@10) */
  for (i=0, rr=w9, cs=w10; i<2; ++i) (*rr++) -= (*cs++);
  /* #17: @6 = dot(@7, @9) */
  w6 = casadi_dot(2, w7, w9);
  /* #18: @8 = sq(@6) */
  w8 = casadi_sq( w6 );
  /* #19: @8 = (@0*@8) */
  w8  = (w0*w8);
  /* #20: @10 = horzcat(@5, @4) */
  rr=w10;
  *rr++ = w5;
  *rr++ = w4;
  /* #21: @10 = @10' */
  /* #22: @11 = dot(@10, @9) */
  w11 = casadi_dot(2, w10, w9);
  /* #23: @12 = sq(@11) */
  w12 = casadi_sq( w11 );
  /* #24: @12 = (@0*@12) */
  w12  = (w0*w12);
  /* #25: @8 = (@8+@12) */
  w8 += w12;
  /* #26: @12 = 6 */
  w12 = 6.;
  /* #27: @13 = input[1][0] */
  casadi_copy(arg[1], 7, w13);
  /* #28: @14 = @13[6] */
  for (rr=(&w14), ss=w13+6; ss!=w13+7; ss+=1) *rr++ = *ss;
  /* #29: @12 = (@12*@14) */
  w12 *= w14;
  /* #30: @8 = (@8-@12) */
  w8 -= w12;
  /* #31: output[0][0] = @8 */
  if (res[0]) res[0][0] = w8;
  /* #32: @13 = zeros(7x1) */
  casadi_clear(w13, 7);
  /* #33: @8 = -6 */
  w8 = -6.;
  /* #34: (@13[6] += @8) */
  for (rr=w13+6, ss=(&w8); rr!=w13+7; rr+=1) *rr += *ss++;
  /* #35: output[1][0] = @13 */
  casadi_copy(w13, 7, res[1]);
  /* #36: @1 = zeros(26x1) */
  casadi_clear(w1, 26);
  /* #37: @6 = (2.*@6) */
  w6 = (2.* w6 );
  /* #38: @6 = (@0*@6) */
  w6  = (w0*w6);
  /* #39: @7 = (@6*@7) */
  for (i=0, rr=w7, cs=w7; i<2; ++i) (*rr++)  = (w6*(*cs++));
  /* #40: @15 = (-@7) */
  for (i=0, rr=w15, cs=w7; i<2; ++i) *rr++ = (- *cs++ );
  /* #41: @11 = (2.*@11) */
  w11 = (2.* w11 );
  /* #42: @0 = (@0*@11) */
  w0 *= w11;
  /* #43: @10 = (@0*@10) */
  for (i=0, rr=w10, cs=w10; i<2; ++i) (*rr++)  = (w0*(*cs++));
  /* #44: @15 = (@15-@10) */
  for (i=0, rr=w15, cs=w10; i<2; ++i) (*rr++) -= (*cs++);
  /* #45: {@11, @8} = vertsplit(@15) */
  w11 = w15[0];
  w8 = w15[1];
  /* #46: (@1[1] += @8) */
  for (rr=w1+1, ss=(&w8); rr!=w1+2; rr+=1) *rr += *ss++;
  /* #47: (@1[0] += @11) */
  for (rr=w1+0, ss=(&w11); rr!=w1+1; rr+=1) *rr += *ss++;
  /* #48: @10 = (@10+@7) */
  for (i=0, rr=w10, cs=w7; i<2; ++i) (*rr++) += (*cs++);
  /* #49: @10 = @10' */
  /* #50: {@11, @8} = horzsplit(@10) */
  w11 = w10[0];
  w8 = w10[1];
  /* #51: @4 = (@4*@8) */
  w4 *= w8;
  /* #52: @5 = (@5*@11) */
  w5 *= w11;
  /* #53: @4 = (@4+@5) */
  w4 += w5;
  /* #54: {@5, NULL} = jac_jac_traj(@2, @3, @3) */
  arg1[0]=(&w2);
  arg1[1]=0;
  arg1[2]=0;
  res1[0]=(&w5);
  res1[1]=0;
  if (casadi_f5(arg1, res1, iw, w, 0)) return 1;
  /* #55: @10 = (@0*@9) */
  for (i=0, rr=w10, cs=w9; i<2; ++i) (*rr++)  = (w0*(*cs++));
  /* #56: @10 = @10' */
  /* #57: {@0, @11} = horzsplit(@10) */
  w0 = w10[0];
  w11 = w10[1];
  /* #58: @9 = (@6*@9) */
  for (i=0, rr=w9, cs=w9; i<2; ++i) (*rr++)  = (w6*(*cs++));
  /* #59: @9 = @9' */
  /* #60: {@6, @8} = horzsplit(@9) */
  w6 = w9[0];
  w8 = w9[1];
  /* #61: @0 = (@0-@8) */
  w0 -= w8;
  /* #62: @5 = (@5*@0) */
  w5 *= w0;
  /* #63: @4 = (@4+@5) */
  w4 += w5;
  /* #64: {@5, NULL} = jac_jac_traj(@2, @3, @3) */
  arg1[0]=(&w2);
  arg1[1]=0;
  arg1[2]=0;
  res1[0]=(&w5);
  res1[1]=0;
  if (casadi_f6(arg1, res1, iw, w, 0)) return 1;
  /* #65: @11 = (@11+@6) */
  w11 += w6;
  /* #66: @5 = (@5*@11) */
  w5 *= w11;
  /* #67: @4 = (@4+@5) */
  w4 += w5;
  /* #68: (@1[12] += @4) */
  for (rr=w1+12, ss=(&w4); rr!=w1+13; rr+=1) *rr += *ss++;
  /* #69: output[1][1] = @1 */
  if (res[1]) casadi_copy(w1, 26, res[1]+7);
  return 0;
}

CASADI_SYMBOL_EXPORT int f1tenth_MJPC_cost_ext_cost_fun_jac(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int f1tenth_MJPC_cost_ext_cost_fun_jac_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int f1tenth_MJPC_cost_ext_cost_fun_jac_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void f1tenth_MJPC_cost_ext_cost_fun_jac_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int f1tenth_MJPC_cost_ext_cost_fun_jac_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void f1tenth_MJPC_cost_ext_cost_fun_jac_release(int mem) {
}

CASADI_SYMBOL_EXPORT void f1tenth_MJPC_cost_ext_cost_fun_jac_incref(void) {
}

CASADI_SYMBOL_EXPORT void f1tenth_MJPC_cost_ext_cost_fun_jac_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int f1tenth_MJPC_cost_ext_cost_fun_jac_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int f1tenth_MJPC_cost_ext_cost_fun_jac_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real f1tenth_MJPC_cost_ext_cost_fun_jac_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* f1tenth_MJPC_cost_ext_cost_fun_jac_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* f1tenth_MJPC_cost_ext_cost_fun_jac_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* f1tenth_MJPC_cost_ext_cost_fun_jac_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s7;
    case 1: return casadi_s8;
    case 2: return casadi_s9;
    case 3: return casadi_s9;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* f1tenth_MJPC_cost_ext_cost_fun_jac_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s10;
    case 1: return casadi_s11;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int f1tenth_MJPC_cost_ext_cost_fun_jac_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 9;
  if (sz_res) *sz_res = 6;
  if (sz_iw) *sz_iw = 8;
  if (sz_w) *sz_w = 62;
  return 0;
}

CASADI_SYMBOL_EXPORT int f1tenth_MJPC_cost_ext_cost_fun_jac_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 9*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 6*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 8*sizeof(casadi_int);
  if (sz_w) *sz_w = 62*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
