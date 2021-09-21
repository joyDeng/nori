#include "nori/shadingPoint.h"
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>

#define THREADDIM 512
// #define ETA 1.5f
// #define MAXNEIGHBOR 100
#define HASH_LITTLE_ENDIAN 1

typedef struct KNNkernelParameters{
    float3 minb;
    float3 cellsize;
    int3 dim;
    int k;
    int dk;
    int maxK;
    float maxRadius;
} KPara;

#define INFINITE std::numeric_limits::infinity();

__constant__ KPara d_params;
__constant__ int d_sizeofsp;
__constant__ int d_grid_size;
__constant__ int d_k;
__constant__ int d_index_offset;
__constant__ int d_num_thread_per_kernel;
__constant__ int d_maxMem;
__constant__ int d_blocknum;
__constant__ int d_iterations;
__constant__ int d_num_clusters;
__constant__ int d_table_size;
__constant__ int d_last_cluster_size;
__constant__ int d_individual;
__constant__ size_t d_pitch_n;
__constant__ int d_num_clusters_point;
__constant__ float d_min_dist;
__device__ int d_debug_cluster_idx;


void CUDAErrorLog(int e, std::string ctr);


/* hash function gridients */

__device__ uint32_t hashsize(uint32_t n) {
    return (uint32_t)1<<(n); 
}
__device__ uint32_t hashmask(uint32_t n) {
    return hashsize(n)-1;
}
__device__ uint32_t rot(uint32_t *x, uint32_t k){
    return ((*x)<<(k)) | ((*x)>>(32-(k)));
}

__device__ void mix(uint32_t *a, uint32_t *b, uint32_t *c)
{
  *a -= *c;  *a ^= rot(c, 4);  *c += *b;
  *b -= *a;  *b ^= rot(a, 6);  *a += *c;
  *c -= *b;  *c ^= rot(b, 8);  *b += *a;
  *a -= *c;  *a ^= rot(c,16);  *c += *b;
  *b -= *a;  *b ^= rot(a,19);  *a += *c;
  *c -= *b;  *c ^= rot(b, 4);  *b += *a;
}

__device__ void final(uint32_t *a, uint32_t *b, uint32_t *c)
{
  *c ^= *b; *c -= rot(b,14);
  *a ^= *c; *a -= rot(c,11);
  *b ^= *a; *b -= rot(a,25);
  *c ^= *b; *c -= rot(b,16);
  *a ^= *c; *a -= rot(c,4); 
  *b ^= *a; *b -= rot(a,14);
  *c ^= *b; *c -= rot(b,24);
}

/*
-------------------------------------------------------------------------------
hashlittle() -- hash a variable-length key into a 32-bit value
  k       : the key (the unaligned variable-length array of bytes)
  length  : the length of the key, counting by bytes
  initval : can be any 4-byte value
Returns a 32-bit value.  Every bit of the key affects every bit of
the return value.  Two keys differing by one or two bits will have
totally different hash values.

The best hash table sizes are powers of 2.  There is no need to do
mod a prime (mod is sooo slow!).  If you need less than 32 bits,
use a bitmask.  For example, if you need only 10 bits, do
  h = (h & hashmask(10));
In which case, the hash table should have hashsize(10) elements.

If you are hashing n strings (uint8_t **)k, do it like this:
  for (i=0, h=0; i<n; ++i) h = hashlittle( k[i], len[i], h);

By Bob Jenkins, 2006.  bob_jenkins@burtleburtle.net.  You may use this
code any way you wish, private, educational, or commercial.  It's free.

Use for hash table lookup, or anything where one collision in 2^^32 is
acceptable.  Do NOT use for cryptographic purposes.
-------------------------------------------------------------------------------
*/

__device__ uint32_t hashlittle( const void *key, size_t length, uint32_t initval)
{
  uint32_t a,b,c;                                          /* internal state */
  union { const void *ptr; size_t i; } u;     /* needed for Mac Powerbook G4 */

  /* Set up the internal state */
  a = b = c = 0xdeadbeef + ((uint32_t)length) + initval;

  u.ptr = key;
  if (HASH_LITTLE_ENDIAN && ((u.i & 0x3) == 0)) {
    const uint32_t *k = (const uint32_t *)key;         /* read 32-bit chunks */
    const uint8_t  *k8;

    /*------ all but last block: aligned reads and affect 32 bits of (a,b,c) */
    while (length > 12)
    {
      a += k[0];
      b += k[1];
      c += k[2];
      mix(&a,&b,&c);
      length -= 12;
      k += 3;
    }

    k8 = (const uint8_t *)k;
    switch(length)
    {
    case 12: c+=k[2]; b+=k[1]; a+=k[0]; break;
    case 11: c+=((uint32_t)k8[10])<<16;  /* fall through */
    case 10: c+=((uint32_t)k8[9])<<8;    /* fall through */
    case 9 : c+=k8[8];                   /* fall through */
    case 8 : b+=k[1]; a+=k[0]; break;
    case 7 : b+=((uint32_t)k8[6])<<16;   /* fall through */
    case 6 : b+=((uint32_t)k8[5])<<8;    /* fall through */
    case 5 : b+=k8[4];                   /* fall through */
    case 4 : a+=k[0]; break;
    case 3 : a+=((uint32_t)k8[2])<<16;   /* fall through */
    case 2 : a+=((uint32_t)k8[1])<<8;    /* fall through */
    case 1 : a+=k8[0]; break;
    case 0 : return c;
    }

    /*----------------------------- handle the last (probably partial) block */
    /* 
     * "k[2]&0xffffff" actually reads beyond the end of the string, but
     * then masks off the part it's not allowed to read.  Because the
     * string is aligned, the masked-off tail is in the same word as the
     * rest of the string.  Every machine with memory protection I've seen
     * does it on word boundaries, so is OK with this.  But VALGRIND will
     * still catch it and complain.  The masking trick does make the hash
     * noticably faster for short strings (like English words).
     */

  } else if (HASH_LITTLE_ENDIAN && ((u.i & 0x1) == 0)) {
    const uint16_t *k = (const uint16_t *)key;         /* read 16-bit chunks */
    const uint8_t  *k8;

    /*--------------- all but last block: aligned reads and different mixing */
    while (length > 12)
    {
      a += k[0] + (((uint32_t)k[1])<<16);
      b += k[2] + (((uint32_t)k[3])<<16);
      c += k[4] + (((uint32_t)k[5])<<16);
      mix(&a,&b,&c);
      length -= 12;
      k += 6;
    }

    /*----------------------------- handle the last (probably partial) block */
    k8 = (const uint8_t *)k;
    switch(length)
    {
    case 12: c+=k[4]+(((uint32_t)k[5])<<16);
             b+=k[2]+(((uint32_t)k[3])<<16);
             a+=k[0]+(((uint32_t)k[1])<<16);
             break;
    case 11: c+=((uint32_t)k8[10])<<16;     /* fall through */
    case 10: c+=k[4];
             b+=k[2]+(((uint32_t)k[3])<<16);
             a+=k[0]+(((uint32_t)k[1])<<16);
             break;
    case 9 : c+=k8[8];                      /* fall through */
    case 8 : b+=k[2]+(((uint32_t)k[3])<<16);
             a+=k[0]+(((uint32_t)k[1])<<16);
             break;
    case 7 : b+=((uint32_t)k8[6])<<16;      /* fall through */
    case 6 : b+=k[2];
             a+=k[0]+(((uint32_t)k[1])<<16);
             break;
    case 5 : b+=k8[4];                      /* fall through */
    case 4 : a+=k[0]+(((uint32_t)k[1])<<16);
             break;
    case 3 : a+=((uint32_t)k8[2])<<16;      /* fall through */
    case 2 : a+=k[0];
             break;
    case 1 : a+=k8[0];
             break;
    case 0 : return c;                     /* zero length requires no mixing */
    }

  } else {                        /* need to read the key one byte at a time */
    const uint8_t *k = (const uint8_t *)key;

    /*--------------- all but the last block: affect some 32 bits of (a,b,c) */
    while (length > 12)
    {
      a += k[0];
      a += ((uint32_t)k[1])<<8;
      a += ((uint32_t)k[2])<<16;
      a += ((uint32_t)k[3])<<24;
      b += k[4];
      b += ((uint32_t)k[5])<<8;
      b += ((uint32_t)k[6])<<16;
      b += ((uint32_t)k[7])<<24;
      c += k[8];
      c += ((uint32_t)k[9])<<8;
      c += ((uint32_t)k[10])<<16;
      c += ((uint32_t)k[11])<<24;
      mix(&a,&b,&c);
      length -= 12;
      k += 12;
    }

    /*-------------------------------- last block: affect all 32 bits of (c) */
    switch(length)                   /* all the case statements fall through */
    {
    case 12: c+=((uint32_t)k[11])<<24;
    case 11: c+=((uint32_t)k[10])<<16;
    case 10: c+=((uint32_t)k[9])<<8;
    case 9 : c+=k[8];
    case 8 : b+=((uint32_t)k[7])<<24;
    case 7 : b+=((uint32_t)k[6])<<16;
    case 6 : b+=((uint32_t)k[5])<<8;
    case 5 : b+=k[4];
    case 4 : a+=((uint32_t)k[3])<<24;
    case 3 : a+=((uint32_t)k[2])<<16;
    case 2 : a+=((uint32_t)k[1])<<8;
    case 1 : a+=k[0];
             break;
    case 0 : return c;
    }
  }

  final(&a,&b,&c);
  return c;
}

/* functions that can only be called by device kernels */
__device__ float dot(const float *a, const float *b){
    float ret = 0.0;
    for(int i = 0 ; i<3 ; i++){
        ret += (a[i] * b[i]);
    }
    float result = fminf(fmaxf(0.0f, ret), 1.0f);
    return result;
}

__device__ void normalize(float *a){
    float temp = 0.0;
    for(int i = 0 ; i<3 ; i++){
        temp += a[i] * a[i];
    }
    temp = sqrtf(temp);
    for (int i = 0 ; i<3 ; i++){
        a[i] /= temp;
    }
}

__device__ void zero3f(float *ret){
    for (int i = 0 ; i<3 ; i++){
        ret[i] = 0.0f;
    }
}

__device__ void one3f(float *ret){
    for (int i = 0 ; i<3 ; i++){
        ret[i] = 1.0f;
    }
}

__device__ void add3f(const float *a, const float *b, float *c){
    for (int i = 0 ; i<3 ; i++){
        c[i] = a[i] + b[i];
    }
}

__device__ void sub3f(const float *a, const float *b, float *c){
    for (int i = 0 ; i<3 ; i++){
        c[i] = a[i] - b[i];
    }
}

__device__ void add3facumulate(float *a, const float *b){
    for (int i = 0 ; i<3 ; i++){
        a[i] += b[i];
    }
}

__device__ void add3fand1f(const float *a, const float v, float *c){
    for (int i = 0 ; i<3 ; i++){
        c[i] = a[i] + v;
    }
}

__device__ void mul3fand1f(const float *a, const float v, float *ret){
    for (int i = 0 ; i<3 ; i++){
        ret[i] = a[i] * v;
    }
}

__device__ void mul3fand3f(const float *a, const float *v, float *ret){
    for (int i = 0 ; i<3 ; i++){
        ret[i] = a[i] * v[i];
    }
}

__device__ float hypot2(const float a, const float b){
    float r;
    if (abs(a) > abs(b)) {
        r = b / a;
        r = abs(a) * sqrtf(1.0f + r*r);
    } else if (b != 0.0f) {
        r = a / b;
        r = abs(b) * sqrtf(1.0f + r*r);
    } else {
        r = 0.0f;
    }
    return r;
}

__device__ float distreval(float cosh, float roughness){
    if(cosh <= 0.0f)
    return 0.0f;
    
    float cosTheta2 = cosh * cosh;
    float beckmannExponent = (1.0 - cosTheta2) / (cosTheta2 * roughness * roughness);
    float root = (1.0f + beckmannExponent) * cosTheta2;
    float result = 1.0f / (M_PI * roughness * roughness * root * root);

    return result;
}



__device__ void fresnelConductorExact(float* result, float cosThetaI, const float* eta, const float* k){
    float cosThetaI2 = cosThetaI*cosThetaI, 
          sinThetaI2 = 1-cosThetaI2,
          sinThetaI4 = sinThetaI2*sinThetaI2;
    
    for (int i = 0 ; i < 3 ; i++){
        float temp1 = eta[i] * eta[i] - k[i] * k[i] - sinThetaI2;
        float a2pb2 = sqrt(temp1 * temp1 + k[i] * k[i] * eta[i] * eta[i] * 4.0);
        float a = sqrt((a2pb2 + temp1) * 0.5f);

        float term1 = a2pb2 + cosThetaI2;
        float term2 = a * (2 * cosThetaI);

        float Rs2 = (term1 - term2) / (term1 + term2);
        float term3 = a2pb2*cosThetaI2 + sinThetaI4;
        float term4 = term2*sinThetaI2;

        float Rp2 = Rs2 * (term3 - term4) / (term3 + term4);
        result[i] = 0.5f * (Rp2 + Rs2);
    }
}

__device__ float fresnelDielectricExt2(float cosThetaI_, float *cosThetaT_, float ETA){
    if(ETA == 1.0f){
        *cosThetaT_ = -cosThetaI_;
        return 0.0f;
    }
    
    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    float scale = (cosThetaI_ > 0.0) ? 1.0 / ETA : ETA;
    float cosThetaTSqr = 1.0 - (1.0-cosThetaI_*cosThetaI_) * (scale*scale);

    /* Check for total internal reflection */
    if (cosThetaTSqr <= 0.0f){
        *cosThetaT_ = 0.0f;
        return 1.0f;
    }
        

    /* Find the absolute cosines of the incident/transmitted rays */
    float cosThetaI = abs(cosThetaI_);
    float cosThetaT = sqrtf(cosThetaTSqr);

    float Rs = (cosThetaI - ETA * cosThetaT)
             / (cosThetaI + ETA * cosThetaT);
    float Rp = (ETA * cosThetaI - cosThetaT)
             / (ETA * cosThetaI + cosThetaT);

    *cosThetaT_ = (cosThetaI_ > 0) ? -cosThetaT : cosThetaT;

    /* No polarization -- return the unpolarized reflectance */
    return 0.5f * (Rs * Rs + Rp * Rp);
}

__device__ float fresnelDielectricExt(float cosThetaI_, float ETA){
    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    float scale = (cosThetaI_ > 0.0) ? 1.0 / ETA : ETA;
    float cosThetaTSqr = 1.0 - (1.0-cosThetaI_*cosThetaI_) * (scale*scale);

    /* Check for total internal reflection */
    if (cosThetaTSqr <= 0.0f)
        return 1.0f;

    /* Find the absolute cosines of the incident/transmitted rays */
    float cosThetaI = abs(cosThetaI_);
    float cosThetaT = sqrtf(cosThetaTSqr);

    float Rs = (cosThetaI - ETA * cosThetaT)
             / (cosThetaI + ETA * cosThetaT);
    float Rp = (ETA * cosThetaI - cosThetaT)
             / (ETA * cosThetaI + cosThetaT);

    /* No polarization -- return the unpolarized reflectance */
    return 0.5f * (Rs * Rs + Rp * Rp);
}

__device__ float smithG1(float dotWH, float dotWShN, float roughness){
    if (dotWH * dotWShN <= 0.0f) return 0.0f;
    float cosWShN2 = dotWShN * dotWShN;
    float tansqrt = (1.0 - cosWShN2) / cosWShN2;
    float tsqrt = sqrtf(tansqrt);
    float tanTheta = abs(tsqrt);
    if (tanTheta == 0.0f) return 1.0f;

    float root = roughness * tanTheta;
    float ret = 2.0f / (1.0f + hypot2(1.0f, root));
    return ret;
}

__device__ float distrpdf(const float dotWoShN, const float dotWoH, const float dotWhShN, const float roughness){
    if (dotWoShN == 0) return 0.0f;
    return  smithG1(dotWoH, dotWoShN, roughness) * abs(dotWoH) * distreval(dotWhShN, roughness) / abs(dotWoShN);
}

__device__ void reflect(float *ret, const float*dir){
    for (int i = 0 ; i < 2; i++)
        ret[i] = -dir[i];
    ret[2] = dir[2];
}

__device__ void refract(float *ret, const float*wo, float costhetaT, float eta){
    float scale = (-costhetaT < 0 ? 1.0 / eta : eta);
    ret[0] = scale * wo[0];
    ret[1] = scale * wo[1];
    ret[2] = costhetaT;
}

/*bsdf evaluation given shading point and incoming light direction */
__device__ void bsdfeval_device(const SPoint sp, const float *wi, float *bsdf, bool t=false){
    float dotWiShN = dot(wi, sp.shN);
    float dotWigeoN = dot(wi, sp.geoN);
    float dotWoGeoN = dot(sp.wo, sp.geoN);
    float dotWoShN = dot(sp.wo, sp.shN);

    if(sp.bsdf_type == 't'){//dielectric
        float cosThetaI = dotWoShN;
        float cosThetaT_ = 0.0f;
        float F =  fresnelDielectricExt2(cosThetaI, &cosThetaT_, sp.eta[0]);
        if(dotWiShN * dotWoShN >= 0.0){
            float dir[3] = {0.0};
            reflect(dir, sp.wo);
            printf("reflect agles %.4f\n", dot(dir, wi)-1.0f);
            if(abs(dot(wi, dir)-1.0f) > 0.00001f)
                return;
            mul3fand1f(sp.specular , F, bsdf);
        }else{
            float dir[3] = {0.0};
            refract(dir, sp.wo, cosThetaT_, sp.eta[0]);
            printf("refract agles %.4f\n", dot(dir, wi)-1.0f);
            if(abs(dot(dir, wi)-1.0f) > 0.00001f)
                return;
            float factor = (cosThetaT_ < 0.0 ? 1.0 / sp.eta[0] : sp.eta[0]);
            // compute transimittance
            printf("specular %.3f, %.3f, %.3f", sp.diffuse[0], sp.diffuse[1], sp.diffuse[2]);
            mul3fand1f(sp.diffuse, factor * factor * (1.0-F), bsdf);
        }
        return;
    }


   
    if( dotWigeoN  * dotWiShN <= 0.0) return;
    if( dotWiShN <= 0.0 || dotWoShN <= 0.0) return;

    float diffuseconst = M_1_PI * dotWiShN;
    float diffuse[3];
    mul3fand1f(sp.diffuse, diffuseconst, diffuse);

    /* compute specular part */
    if(sp.bsdf_type == 'd'){//diffuse
        
        bsdf[0] = diffuse[0];
        bsdf[1] = diffuse[1];
        bsdf[2] = diffuse[2];
    }else if(sp.bsdf_type == 'o'){//opaque
        float wh[3];
        add3f(wi, sp.wo, wh);
        normalize(wh);
        float dotWhShN = dot(wh, sp.shN);
        float D = distreval(dotWhShN, sp.roughness);

        float dotWoH = dot(sp.wo, wh);
        float dotWiH = dot(wi, wh);
        float F = fresnelDielectricExt(dotWoH, 1.5);
        float G = smithG1(dotWoH, dotWoShN, sp.roughness) * smithG1(dotWiH, dotWiShN, sp.roughness);

        float specularconst = F * G * D / (4.0f * dotWoShN);
        float specular[3];
        mul3fand1f(sp.specular, specularconst, specular);

        /*energy conservation fix*/
        // float dotWoShN = dot(sp.shN, sp.wo);
        float T1221 = (1.0 - fresnelDielectricExt(dotWoShN, 1.5)) * (1.0 - fresnelDielectricExt(dotWiShN, 1.5));
        mul3fand1f(diffuse, T1221, diffuse);
        // if(t){
        //     printf("T1221 = %.3f, F = %.5f, D = %.5f, G = %.5f, diffuse = [%.4f, %.4f, %.4f], cosh = %.9f\n", T1221, F, D, G, diffuse[0], diffuse[1], diffuse[2], dotWhShN);
        //     printf("wh = [%.4f, %.4f, %.4f], shN = [%.4f, %.4f, %.4f]\n", wh[0], wh[1], wh[2], sp.shN[0], sp.shN[1], sp.shN[2]);
        //     printf("wi = [%.4f, %.4f, %.4f], wo = [%.4f, %.4f, %.4f]\n", wi[0], wi[1], wi[2], sp.wo[0], sp.wo[1], sp.wo[2]);
        // }

        /*compute diffuse part*/
        add3f(diffuse, specular, bsdf);
    }else if(sp.bsdf_type == 'c'){//rough conductor
        float wh[3];
        add3f(wi, sp.wo, wh);
        normalize(wh);
        float dotWhShN = dot(wh, sp.shN);
        float D = distreval(dotWhShN, sp.roughness);
        if(D == 0.0)
            return; //bsdf = 0.0f
        
        float dotWoH = dot(sp.wo, wh);
        float dotWiH = dot(wi, wh);
        float G = smithG1(dotWoH, dotWoShN, sp.roughness) * smithG1(dotWiH, dotWiShN, sp.roughness);
        float model = D * G / (4.0f * dotWoShN);
        float F[3] = {1.0f};
        float cosThetaI = dotWoH;
        fresnelConductorExact(F, cosThetaI, sp.eta, sp.k);
        // printf("eta = %f, %f, %f\n", sp.eta[0], sp.eta[1], sp.eta[2]);
        // printf("k = %f, %f, %f\n", sp.k[0], sp.k[1], sp.k[2]);
        mul3fand3f(F, sp.specular, F);
        mul3fand1f(F, model, bsdf);
    }
}

/*pdf evaluation given shading point and incoming light direction */
__device__ void pdf_device(const SPoint sp, const float *wi, float *pdf){
    float dotWiShN = dot(wi, sp.shN);
    float dotWigeoN = dot(wi, sp.geoN);
    float dotWoGeoN = dot(sp.wo, sp.geoN);
    float dotWoShN = dot(sp.wo, sp.shN);
    
    if(sp.bsdf_type == 't'){//dielectric
        float cosThetaI = dotWoShN;
        float cosThetaT_;
        float F =  fresnelDielectricExt2(cosThetaI, &cosThetaT_, sp.eta[0]);
        if(dotWiShN * dotWoShN >= 0.0){
            float dir[3] = {0.0f};
            reflect(dir, sp.wo);
            if(abs(dot(dir, wi) - 1.0f) > 0.00001f){
                *pdf = 0.0f;
                return;
            }
            *pdf = F;
        }else{
            float dir[3] = {0.0f};
            refract(dir, sp.wo, cosThetaT_, sp.eta[0]);
            if(abs(dot(dir, wi) - 1.0f) > 0.00001f){
                *pdf = 0.0f;
                return;
            }
            *pdf = 1.0 - F;
        }
        return;
    }

    if( dotWigeoN  * dotWiShN <= 0.0) return;
    if( dotWiShN <= 0.0 || dotWoShN <= 0.0) return;

    float diffuse = dotWiShN * M_1_PI;
    if(sp.bsdf_type == 'd'){
        // printf("is diffuse surface!");
        *pdf = diffuse;
    }else if(sp.bsdf_type == 'o'){
        float pspecular = fresnelDielectricExt(dotWoShN, 1.5);
        float pdiffuse = fmaxf(fmaxf(sp.diffuse[0], sp.diffuse[1]), sp.diffuse[2]);

        pspecular = pspecular / (pspecular + pdiffuse);
        pdiffuse = 1.0 - pspecular;

        float wh[3];
        add3f(wi, sp.wo, wh);
        normalize(wh);
        float dotWhShN = dot(wh, sp.shN);
        float dotWiH = dot(wi, wh);
        float dotWoH = dot(sp.wo, wh);
        float inv_dWhWi = 1.0 / (4.0 * dotWiH);
        float prob = distrpdf(dotWoShN, dotWoH, dotWhShN, sp.roughness);
        float specular = prob * inv_dWhWi * pspecular;
        *pdf = specular + diffuse * pdiffuse;
    }else if(sp.bsdf_type == 'c'){
        float wh[3];
        add3f(wi, sp.wo, wh);
        normalize(wh);
        
        float dotWhShN = dot(wh, sp.shN);
        float dotWiH = dot(wi, wh);
        float dotWoH = dot(sp.wo, wh);
        float inv_whwi = 1.0 / (4.0 * dotWiH);
        float prob = distrpdf(dotWoShN, dotWoH, dotWhShN, sp.roughness);
        *pdf = prob * inv_whwi;
    }
}

__device__ void getKeys(const float *xyz, int3 *keys){
    keys->x = max(0, min((int)floor((xyz[0] - d_params.minb.x) / d_params.cellsize.x), d_params.dim.x-1));
    keys->y = max(0, min((int)floor((xyz[1] - d_params.minb.y) / d_params.cellsize.y), d_params.dim.y-1));
    keys->z = max(0, min((int)floor((xyz[2] - d_params.minb.z) / d_params.cellsize.z), d_params.dim.z-1));
}

__device__ void getKeysAndDirs(float3 xyz, int3 *keys, int3 *dirs){
    keys->x = max(0, min((int)floor((xyz.x - d_params.minb.x) / d_params.cellsize.x), d_params.dim.x-1));
    keys->y = max(0, min((int)floor((xyz.y - d_params.minb.y) / d_params.cellsize.y), d_params.dim.y-1));
    keys->z = max(0, min((int)floor((xyz.z - d_params.minb.z) / d_params.cellsize.z), d_params.dim.z-1));

    float t = xyz.x - d_params.cellsize.x * keys->x - d_params.minb.x;
    if(t > (d_params.cellsize.x / 2.0)){
        dirs->x = 1;
    }else{
        dirs->x = -1;
    }

    t = xyz.y - d_params.cellsize.y * keys->y - d_params.minb.y;
    if(t > (d_params.cellsize.y / 2.0)){
        dirs->y = 1;
    }else{
        dirs->y = -1;
    }

    t = xyz.z - d_params.cellsize.z * keys->z - d_params.minb.z;
    if(t > (d_params.cellsize.z / 2.0)){
        dirs->z = 1;
    }else{
        dirs->z = -1;
    }
}

__device__ int getKey(const int3 keys){
    return d_params.dim.z * d_params.dim.y * keys.x + d_params.dim.z * keys.y + keys.z;
}

__device__ float dev_distance(const float *a, const float *b){
    float dist = 0.0;
    for( int i = 0 ; i < 3 ; i++){
        dist += (float)(a[i] - b[i]) * (float)(a[i] - b[i]);
    }
    dist = sqrtf(dist);
    return dist;
}

/* insert sort */
__device__ void sort(float *dists, int *index, int k){
    float temp_dist;
    int temp_index;
    for (int i = 1; i < k ; i++){
        int j = i;
        while(j>=1 && dists[j] < dists[j-1]){
            temp_index = index[j];
            temp_dist = dists[j];
            dists[j] = dists[j-1];
            index[j] = index[j-1];
            dists[j-1] = temp_dist;
            index[j-1] = temp_index;
            j--;
        }
    }
}

__device__ void sortLast(float *dists, int *index){
    float temp_dist;
    int temp_index;
    int j = d_params.k-1;
    while(j>=1 && dists[j] < dists[j-1]){
        temp_index = index[j];
        temp_dist = dists[j];
        dists[j] = dists[j-1];
        index[j] = index[j-1];
        dists[j-1] = temp_dist;
        index[j-1] = temp_index;
        j--;
    }
}

__device__ int partition(float *dists, int *index, int p, int low, int high){
    float pivot = dists[p];
    int i = low-1;
    int j = high+1;
    float tempd;
    int tempi;
    
    while(1){
        do{
            i+=1;
            if(i > high){
                printf("\nexit bounce: i=%d, j=%d, high=%d, low=%d, p=%d, pivot=%.3f, dists[p]=%.3f, dists[j]=%.3f, dists[j-1]=%.3f, dists[i]=%.3f, dists[i-1]=%.3f, dists[p+1]=%.3f", i, j, high, low, p, pivot, dists[p], dists[j], dists[j-1], dists[i], dists[i-1], dists[p+1]);
            }
        }while(dists[i] < pivot);
        do{
            j-=1;
        }while(dists[j] > pivot);
        if(i >= j) return j;
        tempd = dists[i];
        dists[i] = dists[j];
        dists[j] = tempd;

        tempi = index[i];
        index[i] = index[j];
        index[j] = tempi;
    }
}


__device__ void copy3f(float *b, const float *a){
    for(int i = 0 ; i < 3 ; i++)
        b[i] = a[i];
}

__device__ void findKSmallest(float *dists, int *index, int low, int high, int k){
    int lb = low;
    int hb = high;
    int p = partition(dists, index, k, lb, hb);
    int flag = 0;
    while(p != k && lb < high){
        if(p > k){
            flag = 1;
            hb = p;
            p = partition(dists, index, k, lb, hb);
        }else{
            flag = 0;
            lb = p;
            p = partition(dists, index, k, lb, hb);
        }
        if(lb==p && dists[lb] == dists[k] && flag == 0) break;
    }
}

__device__ void checkOneDim(int *l, int *h, int dim){
    if(*l == 0 && (*h - *l < 1) && *h < dim - 1) {
        (*h) += 1;
        return;
    }
    if(*h == dim - 1 && (*h - *l < 1) && (*l > 0)){
        (*l) -= 1;
        return;
    }
}

__device__ void makesureGrid(int3 *lb , int3 *hb){
    checkOneDim(&lb->x, &hb->x, d_params.dim.x);
    checkOneDim(&lb->y, &hb->y, d_params.dim.y);
    checkOneDim(&lb->z, &hb->z, d_params.dim.z);
}

__device__ float guassian(float dist){
   return  expf(-0.5 * dist * dist);
}

__device__ void findSmallest(float *dist, int *index, int low, int high){
    int temp_index;
    float temp_dist;
    for(int i = high ; i > low ; i--){
        if (dist[i] < dist[i-1]){
            temp_dist = dist[i];
            temp_index = index[i];
            dist[i] = dist[i-1];
            index[i] = index[i-1];
            dist[i-1] = temp_dist;
            index[i-1] = temp_index;
        }
    }
}

__device__ uint32_t hashbmap(uint32_t idx){
    return (idx *  5039 + 39916801) % d_table_size;
}

__device__ uint32_t hashburtle(uint32_t idx){
    uint32_t c = hashlittle(&idx, sizeof(uint32_t), 1) % d_table_size;
    return c;
}

__device__ void swap(int *a, int *b){
    int c = *a;
    *a = *b;
    *b = c;
}

__global__ void addEmitterToDirectLight(const LPoint *lps, val3f *direct_radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    add3f(direct_radiance[Index].value, lps[Index].L_em, direct_radiance[Index].value);
}

__device__ int pointInCell(const int *cinfo, const int offset, const int idx){
    if (idx == d_grid_size- 1){ // if is the last grid
        return d_sizeofsp - offset;
    }else{
        return cinfo[idx+1] - offset;
    }
}

__device__ int getCellSize(const int *array, const int idx, const int size, const int max){
    if (idx == size - 1){
        return max - array[idx];
    }else{
        return array[idx+1] - array[idx];
    }
}

__device__ int getCellSize2(const size_t *array, const int idx, const int size, const size_t max){
    if (idx == size - 1){
        return max - array[idx];
    }else{
        return array[idx+1] - array[idx];
    }
}
/* cluster function */
__global__ void HashGridSize(const SPoint *sps, int *num_point_in_hashgrids){
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= d_sizeofsp) return;
    int3 xyzs = make_int3(0,0,0);
    getKeys(sps[index].pos, &xyzs);
    uint32_t key = getKey(xyzs);
    atomicAdd(&num_point_in_hashgrids[key], 1);
}

__global__ void HashTableSize(const SPoint *sps, const int *cu_clusters_idx, int *num_point_in_hashtable){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_num_clusters) return;
    uint32_t spidx = cu_clusters_idx[idx];
    int3 xyzs = make_int3(0,0,0);
    getKeys(sps[spidx].pos, &xyzs);
    uint32_t key = hashbmap(getKey(xyzs));
    atomicAdd(&num_point_in_hashtable[key], 1);
}

__global__ void getOffset(const SPoint *sps, int *p_offset, const int *cu_offsets){
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= d_sizeofsp) return;
    int3 xyzs = make_int3(0,0,0);
    getKeys(sps[index].pos, &xyzs);
    uint32_t key = getKey(xyzs);
    p_offset[index] = cu_offsets[key];
}

__global__ void buildHash(const SPoint *sps, const int *offset, int *cu_num_in_grid, GridEntry *grids){
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= d_sizeofsp) return;

    SPoint sp = sps[index];
    int offset_g = offset[index];
    int3 xyzs = make_int3(0,0,0);
    getKeys(sp.pos, &xyzs);
    uint32_t key = getKey(xyzs);
   
    uint32_t idx_in_grid = atomicAdd(&cu_num_in_grid[key], 1);
    grids[offset_g + idx_in_grid].position[0] = sp.pos[0];
    grids[offset_g + idx_in_grid].position[1] = sp.pos[1];
    grids[offset_g + idx_in_grid].position[2] = sp.pos[2];
    grids[offset_g + idx_in_grid].index = index;
}

__global__ void buildHashSub(const SPoint *sps, const int * cu_cluster_idx, const int *cu_hash_offset, int *cu_num_in_grid, GridEntry *cu_hash_table){
    uint32_t Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_num_clusters) return;
    int spidx = cu_cluster_idx[Index];

    SPoint sp = sps[spidx];
    int3 xyzs = make_int3(0,0,0);
    getKeys(sp.pos, &xyzs);
    uint32_t key = hashbmap(getKey(xyzs));

    int hash_offset = cu_hash_offset[key];
    int idx_in_grid = atomicAdd(&cu_num_in_grid[key], 1);
    // if (npingrid == 0)
    //     printf("WARNING!!!!!!!!!!!!!!!!!!!!! idx_in_grid = %d, hash_offset = %d, spidx = %d", idx_in_grid, hash_offset, spidx);
    // cu_hash_table[offset_g + idx_in_grid].position[0] = sp.pos[0];
    // cu_hash_table[offset_g + idx_in_grid].position[1] = sp.pos[1];
    // cu_hash_table[offset_g + idx_in_grid].position[2] = sp.pos[2];
    copy3f(cu_hash_table[hash_offset + idx_in_grid].position, sp.pos);
    cu_hash_table[hash_offset + idx_in_grid].index = Index;
}

__global__ void countClusters(int * cu_num_in_cell, int * cu_num_small_cluster, int K){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= d_num_clusters) return;
    int num_small_clusters = cu_num_in_cell[index] / K + 1;
    if (num_small_clusters > 1){
        cu_num_small_cluster[index] = num_small_clusters;
    }else {
        cu_num_small_cluster[index] = 1;
    }
}

// __global__ void updateClusterIdx(SPoint *cu_sps, 
//                                 const int *cu_num_small_clusters, 
//                                 const int newNumClusters, 
//                                 int *cu_count
//                                 )
// {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if(index >= d_sizeofsp) return;

//     SPoint sp = cu_sps[index];
//     if(sp.groupIdx != -1){
//         int num_small_cluster_offset = cu_num_small_clusters[index];
//         int pnum = getCellSize(cu_num_small_clusters, index, d_num_clusters, newNumClusters);
//         sp.groupIdx = num_small_cluster_offset;
//         atomicAdd(&cu_clusters_sub1[sp.groupIdx],1);
//     }else{
//         sp.groupIdx += newNumClusters;
//     }
// }


__global__ void SubdivideClusters(SPoint *cu_sps, 
                                    int  *cu_clusters, // legnth: sizeofsp
                                    const int * cu_clusters_old_offset, // length: sizeof old cluster
                                    int * cu_num_point_new_cluster, // length: sizeof new cluster
                                    const int * cu_num_small_clusters, // length size of old cluster
                                    const int newNumClusters){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    SPoint sp = cu_sps[Index];
    int cluster_idx = sp.groupIdx;// old cluster idx;
    if (cluster_idx != -1){
        int pnum = getCellSize(cu_num_small_clusters, cluster_idx, d_num_clusters, newNumClusters);
        int cluster_offset = cu_clusters_old_offset[cluster_idx];
        int newgroupIdx = cu_num_small_clusters[cluster_idx];

        cu_sps[Index].groupIdx = newgroupIdx;
        if(pnum > 1){
            float mindist = 262144.0;
            int minidx = 0;
            for(int i = 0 ; i < pnum; i++){
                SPoint clusterpoint = cu_sps[cu_clusters[cluster_offset+i]];
                float dist = dev_distance(clusterpoint.pos, sp.pos);
                if(dist < mindist){
                    mindist = dist;
                    minidx = i;
                }
            }
            cu_sps[Index].groupIdx += minidx;
        }
        atomicAdd(&cu_num_point_new_cluster[cu_sps[Index].groupIdx], 1);
    }
}

__global__ void CheckClusters(const int *cu_neighbors_offset, int *cu_num_in_cell){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_num_clusters) return;

    int pnum = getCellSize(cu_neighbors_offset, Index, d_num_clusters, d_num_clusters_point);
    int currentIdx = cu_num_in_cell[Index];
    if (currentIdx != pnum)
        printf("\n %d th currentIdx = %d, pnum = %d", Index, currentIdx, pnum);
}


__global__ void SaveClusters(SPoint *sps, const int * cu_neighbors_offset, int * cu_num_in_cell, int * cu_clusters){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    int cluster_idx = sps[Index].groupIdx;
    if(cluster_idx != -1){
        int offset = cu_neighbors_offset[cluster_idx];
        // int pnum = getCellSize(cu_neighbors_offset, cluster_idx, d_num_clusters, d_num_clusters_point);
        int currentIdx = atomicAdd(&cu_num_in_cell[cluster_idx], 1);
        // if(currentIdx == pnum-1)
            // printf("\n %d th currentIdx = %d, pnum = %d", Index, currentIdx, pnum);
        cu_clusters[offset + currentIdx] = Index;
    } 
    // else {
        // deal with the point that didn't find any neighbors in 8 grids
        // int offset = d_num_clusters_point;
        // int currentIdx = atomicAdd(&cu_num_in_cell[d_num_clusters], 1);
        // cu_clusters[offset + currentIdx] = Index;
    // }
}

__global__ void Cluster(SPoint *sps, const int * cu_hash_offset, const GridEntry * cu_hash_table, int *  cu_neighbors_sizes){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;
    // compute hash table Index
    SPoint sp = sps[Index];
    
    /*find the 8 grid for searching*/
    int3 keys = make_int3(0,0,0);
    int3 dirs = make_int3(0,0,0);
    float3 mypos = make_float3(sp.pos[0], sp.pos[1], sp.pos[2]);
    getKeysAndDirs(mypos, &keys, &dirs);
    int3 lbb = make_int3(max(0, min(keys.x, keys.x + dirs.x)),max(0, min(keys.y, keys.y + dirs.y)),max(0, min(keys.z, keys.z + dirs.z)));
    int3 rtf = make_int3(min(d_params.dim.x-1, max(keys.x, keys.x + dirs.x)),min(d_params.dim.y-1, max(keys.y, keys.y + dirs.y)),min(d_params.dim.z-1, max(keys.z, keys.z + dirs.z)));
    makesureGrid(&lbb, &rtf);

    float mindistance = d_min_dist;
    int nearest_cluster_id = -1;
    for(int ix = lbb.x ; ix <= rtf.x ; ix++)
        for(int iy = lbb.y ; iy <= rtf.y ; iy++)
            for(int iz = lbb.z ; iz <= rtf.z ; iz++){
                uint32_t tempkey = hashbmap(getKey(make_int3(ix, iy, iz)));
                int offset = cu_hash_offset[tempkey];
                int pnum = getCellSize(cu_hash_offset, tempkey, d_table_size, d_num_clusters);
                for (int i = 0 ; i < pnum; i++){
                    GridEntry G = cu_hash_table[offset + i];
                    float dist = dev_distance(G.position, sp.pos);
                    if (dist <= mindistance){
                        mindistance = dist;
                        nearest_cluster_id = G.index;
                    }
                }
            }
    
    sps[Index].groupIdx = nearest_cluster_id;
    if (nearest_cluster_id != -1)
        atomicAdd(&cu_neighbors_sizes[nearest_cluster_id], 1);
}

__global__ void CountNoneZeroElements(const SPoint *cu_sps, const int *cu_offset, size_t *cu_numberofNeighbor){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= d_sizeofsp) return;

    SPoint sp = cu_sps[index];
    int clusterIdx = sp.groupIdx;
    if(clusterIdx != -1){
        cu_numberofNeighbor[index] = getCellSize(cu_offset, sp.groupIdx, d_num_clusters, d_num_clusters_point);
    }else{
        cu_numberofNeighbor[index] = 1;
    }
}

__global__ void computeNoneZeroElements(const SPoint *sps,const int *cu_clusters, const int *cu_cluster_offset, const float* pdfmarginal, const size_t *cu_element_offset, val3f *cu_matrix_elements){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= d_sizeofsp) return;

    SPoint sp = sps[index];
    int cluster_idx = sp.groupIdx;
    size_t my_element_offset = cu_element_offset[index];
    if (pdfmarginal[index] > 0.0 && sp.nidx > 0 && sp.rrpdf > 1e-7f){
        float inv_pdfmarginal = 1.0 / pdfmarginal[index];
        if (cluster_idx != -1){
            int point_num_in_cluster = getCellSize(cu_cluster_offset, cluster_idx, d_num_clusters, d_num_clusters_point);
            int my_cluster_offset = cu_cluster_offset[cluster_idx];
            
            for ( int i = 0 ; i < point_num_in_cluster ; i++){
                int idx = cu_clusters[i + my_cluster_offset];
                SPoint spo = sps[idx];
                float bsdf[3] = {0.0};
                bsdfeval_device(spo, sp.wi, bsdf);
                mul3fand1f(bsdf,  inv_pdfmarginal, cu_matrix_elements[my_element_offset+i].value);
            }
        }else{
            float bsdf[3] = {0.0};
            bsdfeval_device(sp, sp.wi, bsdf);
            mul3fand1f(bsdf,  inv_pdfmarginal, cu_matrix_elements[my_element_offset].value);
        }
    }   

}
    
/* kernel functions */
__global__ void batchNearestNeighbor(const SPoint *sps, const GridEntry * grids, const int *cinfo, float *dists, int *inds, int *ret, int *cu_knn, int *cu_dknn, size_t pitch_d, size_t pitch_i){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= d_num_thread_per_kernel) return;
    int Index = index + d_index_offset;
    if(Index >= d_sizeofsp) return;

    /*compute the starting point of the indices*/
    // size_t arrayoffset = index * d_maxMem;
    int *my_index_array = (int *)((char *)inds + index * pitch_i);
    float *my_distances_array = (float *)((char *)dists + index * pitch_d);
    int *result = (int *)((char *)ret + Index * d_pitch_n);

    /* get the key of the current grid */
    SPoint sp = sps[Index];
    int3 keys = make_int3(0,0,0);
    int3 dirs = make_int3(0, 0, 0);
    float3 mypos = make_float3(sp.pos[0], sp.pos[1], sp.pos[2]);
    getKeysAndDirs(mypos, &keys, &dirs);
    

    int3 lbb = make_int3(max(0, min(keys.x, keys.x + dirs.x)),max(0, min(keys.y, keys.y + dirs.y)),max(0, min(keys.z, keys.z + dirs.z)));
    int3 rtf = make_int3(min(d_params.dim.x-1, max(keys.x, keys.x + dirs.x)),min(d_params.dim.y-1, max(keys.y, keys.y + dirs.y)),min(d_params.dim.z-1, max(keys.z, keys.z + dirs.z)));

    makesureGrid(&lbb, &rtf);

    /* compute distance for all indices in range, insert based on  */
    /* loop through the neighbor cells, range should be 1 or 2 for speed */
    int total_points = 0;
  
    for(int ix = lbb.x ; ix <= rtf.x ; ix++)
        for(int iy = lbb.y ; iy <= rtf.y ; iy++)
            for(int iz = lbb.z ; iz <= rtf.z ; iz++){
                int3 temkeys = make_int3(ix, iy, iz);
                int tk = getKey(temkeys);
                int offset = cinfo[tk];
                int pnum = pointInCell(cinfo, offset, tk);
                for (int i = 0 ; i < pnum; i++){
                    GridEntry G = grids[offset + i];
                    float dist = dev_distance(G.position, sp.pos);
                    my_distances_array[total_points] = dist;
                    my_index_array[total_points] = G.index;
                    total_points += 1;
                }
            }
    
    int maxK = max(d_params.dk, d_params.k);
    int minK = min(d_params.dk, d_params.k);

    // quicksort(distances, indices, 0, total_points-1);
    if (maxK < total_points){ // incase we didn't get enough points
        // printf("Block Id %d, thread Id %d, total_points = %d, k=%d", blockIdx.x, threadIdx.x, total_points, d_params.k);
        findKSmallest(my_distances_array, my_index_array, 0, total_points-1,maxK-1);
        findKSmallest(my_distances_array, my_index_array, 0, maxK-1, minK-1);
        for( int i = 0 ; i < maxK ; i++)
            result[i] = my_index_array[i];
        cu_knn[Index] = d_params.k;
        cu_dknn[Index] = d_params.dk;
    }else if(minK < total_points){
        findKSmallest(my_distances_array, my_index_array, 0, total_points-1, minK-1);
        for(int i = 0 ; i < total_points ;i++)
            result[i] = my_index_array[i];
        if(d_params.dk > d_params.k){
            cu_knn[Index] = d_params.k;
            cu_dknn[Index] = total_points;
        }else{
            cu_knn[Index] = total_points;
            cu_dknn[Index] = d_params.dk;
        }
        
    }else{
        for(int i = 0 ; i < total_points ;i++)
            result[i] = my_index_array[i];
        cu_knn[Index] = total_points;
        cu_dknn[Index] = total_points;
    }

    // ensure the first point is myself
    // if (d_params.k == 1 || d_params.dk == 1){
        for(int i = 0 ; i < min(maxK, total_points) ; i++){
            if(Index == result[i])  {
                result[i] = result[0];
                result[0] = Index;
                break;
            }
        }
    // }
}




/* kernel functions */
__global__ void NearestNeighbor(const SPoint *sps, const GridEntry * grids, const int *cinfo, float *dists, int *ret, int *cu_knn){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;
    int current = 0;

    /*compute the starting point of the indices*/
    int *indices = &ret[Index * d_params.k];
    float *distances = &dists[Index * d_params.k];

    /* get the key of the current grid */
    SPoint sp = sps[Index];
    int3 keys = make_int3(0,0,0);
    getKeys(sp.pos, &keys);

    int3 lbb = make_int3(max(0, keys.x-1),max(0, keys.y-1),max(0, keys.z-1));
    int3 rtf = make_int3(min(d_params.dim.x-1, keys.x+1),min(d_params.dim.y-1, keys.y+1),min(d_params.dim.z-1, keys.z+1));
    
    makesureGrid(&lbb, &rtf);
    /* compute distance for all indices in range, insert based on  */
    /* loop through the neighbor cells, range should be 1 or 2 for speed */
    for(int ix = lbb.x ; ix <= rtf.x ; ix++)
        for(int iy = lbb.y ; iy <= rtf.y ; iy++)
            for(int iz = lbb.z ; iz <= rtf.z ; iz++){
                int3 temkeys = make_int3(ix, iy, iz);
                // Cell temp_info = cinfo[getKey(temkeys)];
                int tk = getKey(temkeys);
                int offset = cinfo[tk];
                int pnum = pointInCell(cinfo, offset, tk);

                for (int i = 0 ; i < pnum ; i++){
                    float dist = dev_distance(grids[offset + i].position, sp.pos);
                    if( current < d_params.k){
                        distances[current] = dist;
                        indices[current] = grids[offset + i].index;
                        current += 1;
                        if (current == d_params.k){
                            sort(dists, indices, d_params.k);
                        }
                    }else{
                        if(dist < distances[d_params.k-1]){
                            distances[d_params.k-1] = dist;
                            indices[d_params.k-1] = grids[offset + i].index;
                            sortLast(distances, indices);
                        }
                    }
                }
            }
    
    // if I didn't get enough neighbor
    if(current < d_params.k){
        // printf("didn't get enough points %d/%d", current, d_params.k);
        cu_knn[Index] = current;
    }else{
        cu_knn[Index] = d_params.k;
    }
}

// (cu_sps, cu_clusters, cu_cluster_offset, cu_pdfmarginal)
__global__ void allGPUClusterPdfMarginal(const SPoint *sps, const int *clusters, const int *cluster_offsets, float * pdfmarginal){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    SPoint sp = sps[Index];
    /* get index of my cluster*/
    int cluster_idx = sp.groupIdx;
    if (sp.groupIdx != -1){
        int offset = cluster_offsets[cluster_idx];
        int point_num_in_cluster = getCellSize(cluster_offsets, cluster_idx, d_num_clusters, d_num_clusters_point);
        float pdfm = 0.0;
        for ( int i = 0 ; i < point_num_in_cluster ; i++){
            SPoint spo = sps[clusters[offset+i]];
            if(spo.nidx == Index) continue;
            float pdf = 0.0;
            pdf_device(spo, sp.wi, &pdf);
            pdfm += (pdf * spo.rrpdf);
        }
        pdfmarginal[Index] = pdfm;
    }else{
        float pdf = 0.0;
        pdf_device(sp, sp.wi, &pdf);
        pdfmarginal[Index] = pdf * sp.rrpdf;
    }
}

__global__ void MX(const SPoint *sps, const int * cu_clusters, const val3f * cu_elements, const int * cu_cluster_offset, const size_t * cu_element_offset, const val3f * cu_tempRadiance, val3f * cu_radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    SPoint sp = sps[Index];
    /* get index of my cluster*/
    int cluster_idx = sp.groupIdx;
    size_t my_element_offset = cu_element_offset[Index];
    if(sp.nidx > 0){
        if (cluster_idx != -1){
            int point_num_in_cluster = getCellSize(cu_cluster_offset, cluster_idx, d_num_clusters, d_num_clusters_point);
            int cluster_offsets = cu_cluster_offset[cluster_idx];
            for ( int i = 0 ; i < point_num_in_cluster ; i++){
                float rad[3] = {0.0f};
                size_t edx = my_element_offset+i;
                int rdx = cu_clusters[cluster_offsets+i];
                mul3fand3f(cu_elements[edx].value, cu_tempRadiance[Index+1].value, rad);
                atomicAdd(cu_radiance[rdx].value, rad[0]);
                atomicAdd(cu_radiance[rdx].value+1, rad[1]);
                atomicAdd(cu_radiance[rdx].value+2, rad[2]);
                if(isnan(rad[0]) || isnan(rad[1]) || isnan(rad[2])){
                    printf("\n \n ISNAN*****************rad = [%.4f, %.4f, %.4f], incoming = [%.4f, %.4f, %.4f], ", rad[0], rad[1], rad[2], cu_tempRadiance[Index+1].value[0], cu_tempRadiance[Index+1].value[1], cu_tempRadiance[Index+1].value[2]);
                }
                // if (rdx == 2978795)
                //     printf("weight %d is [%.4f, %.4f, %.4f], rad is [%.4f, %.4f, %.4f]\n", i,
                //         cu_elements[edx].value[0], cu_elements[edx].value[1], cu_elements[edx].value[2],
                //         cu_tempRadiance[Index+1].value[0], cu_tempRadiance[Index+1].value[1], cu_tempRadiance[Index+1].value[2]);
            }
            
        }else{
            float rad[3] = {0.0f};
            mul3fand3f(cu_elements[my_element_offset].value, cu_tempRadiance[Index+1].value, rad);
            atomicAdd(cu_radiance[Index].value, rad[0]);
            atomicAdd(cu_radiance[Index].value+1, rad[1]);
            atomicAdd(cu_radiance[Index].value+2, rad[2]);
            if (Index == 2978795)
                printf("weight %d is [%.4f, %.4f, %.4f], rad is [%.4f, %.4f, %.4f]", 1,
                    cu_elements[my_element_offset].value[0], cu_elements[my_element_offset].value[1], cu_elements[my_element_offset].value[2],
                    cu_tempRadiance[Index+1].value[0], cu_tempRadiance[Index+1].value[1], cu_tempRadiance[Index+1].value[2]);
            return;
        }   
    } 
}

__global__ void allGPUClusterScatterRadiance(const SPoint *sps, const float *pdfmarginal, const int *clusters, const int *cu_cluster_offset, const val3f *tempradiance, val3f *radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;
    SPoint sp = sps[Index];
    /* get index of my cluster*/
    int cluster_idx = sp.groupIdx;
    /* check whether this point belong to any cluster, if not do nothing */
    if (pdfmarginal[Index] > 0.0 && sp.nidx > 0 && sp.rrpdf > 1e-7f){
        float inv_pdfmarginal = 1.0 / pdfmarginal[Index];
        if (cluster_idx != -1){
            int point_num_in_cluster = getCellSize(cu_cluster_offset, cluster_idx, d_num_clusters, d_num_clusters_point);
            int my_cluster_offset = cu_cluster_offset[cluster_idx];
            for ( int i = 0 ; i < point_num_in_cluster ; i++){
                int idx = clusters[i + my_cluster_offset];
                if(idx == Index+1) continue;
                SPoint spo = sps[idx];
                float bsdf[3] = {0.0};
                bsdfeval_device(spo, sp.wi, bsdf);
                float rad[3] = {0.0f};
                mul3fand3f(bsdf, tempradiance[Index+1].value, rad);
                mul3fand1f(rad,  inv_pdfmarginal, rad);

                atomicAdd(radiance[idx].value, rad[0]);
                atomicAdd(radiance[idx].value+1, rad[1]);
                atomicAdd(radiance[idx].value+2, rad[2]);
            }
        }else{
            float bsdf[3] = {0.0};
            bsdfeval_device(sp, sp.wi, bsdf);
            float rad[3] = {0.0f};
            mul3fand3f(bsdf, tempradiance[Index+1].value, rad);
            mul3fand1f(rad,  inv_pdfmarginal, radiance[Index].value);
        }
    }    
}


__global__ void allGPUMISRadiance(const SPoint *sps,const float *pdfsums, const int *neighbors, const int *knn, const val3f *tempradiance, val3f *radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    int k = knn[Index];
    int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);
    SPoint sp = sps[Index];

    for ( int i = 0 ; i < k ; i++){
        int neighborIdx = neighborhood[i];
        SPoint spn = sps[neighborIdx];
        /* if has continuous point on the path */
        if(spn.nidx > 0 && pdfsums[neighboroffset+i] > 0.0 && spn.rrpdf > 1e-7f){
            float bsdf[3] = {0.0f};
            bsdfeval_device(sp, spn.wi, bsdf);
            float rad[3] = {0.0f};
            float inv_pdf_sum = 1.0 / pdfsums[neighboroffset+i];
            // mul3fand1f(tempradiance[neighborIdx+1].value, 1.0 / spn.rrpdf, rad);
            mul3fand3f(bsdf, tempradiance[neighborIdx+1].value, rad);
            mul3fand1f(rad, inv_pdf_sum, rad);
            add3facumulate(radiance[Index].value, rad);    
        }
    }
}

__global__ void allGPUMISRadianceJitter(const SPoint *sps,const float *pdfsums, const int *neighbors, const int *knn, const val3f *tempradiance, val3f *radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    int k = knn[Index];
    int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);
    SPoint sp = sps[Index];

    for ( int i = 0 ; i < k ; i++){
        int neighborIdx = neighborhood[i];
        SPoint spn = sps[neighborIdx];
        /* if has continuous point on the path */
        if(spn.groupIdx == sp.groupIdx && spn.nidx > 0 && pdfsums[neighboroffset+i] > 0.0 && spn.rrpdf > 1e-7f){
            float bsdf[3] = {0.0f};
            bsdfeval_device(sp, spn.wi, bsdf);
            float rad[3] = {0.0f};
            float inv_pdf_sum = 1.0 / pdfsums[neighboroffset+i];
            // mul3fand1f(tempradiance[neighborIdx+1].value, 1.0 / spn.rrpdf, rad);
            mul3fand3f(bsdf, tempradiance[neighborIdx+1].value, rad);
            mul3fand1f(rad, inv_pdf_sum, rad);
            add3facumulate(radiance[Index].value, rad);    
        }
    }
}

__global__ void allGPUScatterRadiance(const SPoint *sps,const float *pdfmarginal, const int *neighbors, const int *knn, const val3f *temp_rad, val3f *radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    int k = knn[Index];
    // int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);
    SPoint sp = sps[Index];
    
    /* if has continuous point on the path */
    if(pdfmarginal[Index] > 0.0 && sp.nidx > 0 && sp.rrpdf > 1e-7f){
        float inv_pdfmarginal = 1.0 / (pdfmarginal[Index]);
        for ( int i = 0 ; i < k ; i++){
            int index = neighborhood[i];
            if(index == Index+1) continue;
            SPoint spo = sps[index];
            float bsdf[3] = {0.0f};
            bsdfeval_device(spo, sp.wi, bsdf);
            float rad[3] = {0.0f};
            // float incoming[3] = {1.0f, 1.0f, 1.0f};
            mul3fand3f(temp_rad[Index+1].value, bsdf, rad);
            // mul3fand3f(incoming, bsdf, rad);
            mul3fand1f(rad,  inv_pdfmarginal, rad);
            atomicAdd(radiance[index].value, rad[0]);
            atomicAdd(radiance[index].value+1, rad[1]);
            atomicAdd(radiance[index].value+2, rad[2]);
            
            // float cpdf = 0.0;
            // pdf_device(spo, sp.wi, &cpdf);
            // float w =  cpdf * 1.0 / pdfmarginal[Index];
            // atomicAdd(&weightsum[index], w);
        }
    }
}

__global__ void printWeightSum(const float *weightsum){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    if(weightsum[Index] != 1.0){
        printf("\t [%.5f]", weightsum[Index]);
    }
}

__global__ void computeWeightSum(float *weightsum, float *count,float *minweight){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;
    // printf("minweight = %.8f \n",minweight[Index]);
    // float w = weightsum[Index] - count[Index] * minweight[Index];
    float w = minweight[Index] * count[Index] - weightsum[Index];
    // if(weightsum[Index] < 1e-6f && weightsum[Index] != 0.0){
    //     printf("\twegiht very small! %.8f, maxweight=%.8f, count=%d\n", weightsum[Index], minweight[Index], count[Index]);
    // }
    if(w == 0.0 || weightsum[Index] < 1e-7f){
        minweight[Index] = 0.0;
    }else{
        weightsum[Index] = count[Index] / w;
    }
}

__global__ void allGPUScatterRadianceWithWeight(const SPoint *sps,const float *pdfmarginal, const int *neighbors, const int *knn, const val3f *temp_rad, val3f *radiance, const float *weightsum, const float *minweight){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    int k = knn[Index];
    // int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);
    SPoint sp = sps[Index];
    
    /* if has continuous point on the path */
    if(pdfmarginal[Index] > 0.0 && sp.nidx > 0 && sp.rrpdf > 1e-7f){
        float inv_pdfmarginal = 1.0 / (pdfmarginal[Index]);
        float weight_s = weightsum[Index];
        float max_dist = minweight[Index];
        for ( int i = 0 ; i < k ; i++){
            int index = neighborhood[i];
            if(index == Index+1) continue;
            SPoint spo = sps[index];
            float bsdf[3] = {0.0f};
            bsdfeval_device(spo, sp.wi, bsdf);
            float rad[3] = {0.0f};
            mul3fand3f(bsdf, temp_rad[Index+1].value, rad);
            mul3fand1f(rad,  inv_pdfmarginal, rad);
            float dist = dev_distance(spo.pos, sp.pos);
            // float weight = weightsum[index] * (guassian(dist) - minweight[index]);
            float weight = (max_dist - 0.8 * dist) * weight_s;
            if(weight_s == 0.0) weight = 1.0;
            if(max_dist < dist) weight = 0.0;
            // add3f(radiance[index].value, rad, radiance[index].value);
            // if (weight < 0.0 || rad[0] < 0.0 || rad[1] < 0.0 || rad[2] < 0.0) {
            //     printf("\t []negative value!, i=%d: w=%.8f, max_dist = %.8f, dist = %.8f, ws = %.8f,\n", i, weight, max_dist, dist, weight_s);
            // }
            atomicAdd(radiance[index].value, rad[0] * weight);
            atomicAdd(radiance[index].value+1, rad[1] * weight);
            atomicAdd(radiance[index].value+2, rad[2] * weight);
        }
    }
}

__global__ void lastRun(const SPoint *sps, const val3f *temp_rad, val3f *radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    SPoint sp = sps[Index];
    float spdf = 0.0;
    pdf_device(sp, sp.wi, &spdf);
    /* if has continuous point on the path */
    if(spdf > 0.0 && sp.nidx > 0){
        float inv_pdf = 1.0 / (spdf * sp.rrpdf);
        float bsdf[3] = {0.0f};
        bool t = false;
        // if(Index == 2978795)
        //     t = true;
        bsdfeval_device(sp, sp.wi, bsdf, t);
        float rad[3] = {0.0f};
        mul3fand3f(temp_rad[Index+1].value, bsdf, rad);
        mul3fand1f(rad, inv_pdf, rad);
        // if (Index == 2978795)
        //     printf("\n nidx = %d, roughness = %.5f,\n, rad = [%.3f, %.3f, %.3f], \n temp = [%.3f, %.3f, %.3f], \n spdf = %.3f, sp.pdf = %.3f, sp.rrpdf = %.3f, inv_pdf = %.3f, bsdf = [%.3f, %.3f, %.3f]", 
        //         sp.nidx, sp.roughness,
                // rad[0], rad[1], rad[2], temp_rad[Index+1].value[0], temp_rad[Index+1].value[1], temp_rad[Index+1].value[2], spdf, sp.pdf, sp.rrpdf, inv_pdf, bsdf[0], bsdf[1], bsdf[2]);
        add3facumulate(radiance[Index].value, rad);
        if(isnan(rad[0]) || isnan(rad[1]) || isnan(rad[2])){
            printf("\n \n last run ISNAN*****************rad = [%.4f, %.4f, %.4f], incoming = [%.4f, %.4f, %.4f], ", rad[0], rad[1], rad[2], temp_rad[Index+1].value[0], temp_rad[Index+1].value[1], temp_rad[Index+1].value[2]);
        }
        
    }
    // add3facumulate(radiance[Index].value, sp.eLd);
}

__global__ void lastRunJitter(const SPoint *sps, const float *pdfmarginal, const int *neighbors, const int *knn, const val3f *temp_rad, val3f *radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    int k = knn[Index];
    // int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);
    SPoint sp = sps[Index];
    
    /* if has continuous point on the path */
    if(pdfmarginal[Index] > 0.0 && sp.nidx > 0 && sp.rrpdf > 1e-7f){
        float inv_pdfmarginal = 1.0 / (pdfmarginal[Index]);
        for ( int i = 0 ; i < k ; i++){
            int index = neighborhood[i];
            if(index == Index+1) continue;
            SPoint spo = sps[index];
            if(sp.groupIdx != spo.groupIdx) continue;
            float bsdf[3] = {0.0f};
            bsdfeval_device(spo, sp.wi, bsdf);
            float rad[3] = {0.0f};
            mul3fand3f(temp_rad[Index+1].value, bsdf, rad);
            mul3fand1f(rad,  inv_pdfmarginal, rad);
            atomicAdd(radiance[index].value, rad[0]);
            atomicAdd(radiance[index].value+1, rad[1]);
            atomicAdd(radiance[index].value+2, rad[2]);
        }
    }
}


__global__ void lastRunRand(const SPoint *sps, const int *neighbors, const int *knn, const val3f *temp_rad, val3f *radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    int k = knn[Index];
    if(k == 0)return;
    int rid = Index % k;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);
    SPoint sp = sps[Index];
    if(sp.pdf > 0.0 && sp.nidx > 0){
        if( rid != 0){
            int idx = neighborhood[rid];
            SPoint my_neighbor = sps[idx];
            float pdf_neighbor = 0.0;
            pdf_device(my_neighbor, sp.wi, &pdf_neighbor);
            //float inv_pdf = 2.0 / (sp.pdf  + pdf_neighbor);
            float inv_pdf = 2.0 / (sp.pdf * sp.rrpdf + pdf_neighbor * my_neighbor.rrpdf);
            float bsdf[3] = {0.0f};
            bsdfeval_device(sp, sp.wi, bsdf);
            float rad[3] = {0.0f};
            mul3fand3f(temp_rad[Index+1].value, bsdf, rad);
            mul3fand1f(rad, inv_pdf, rad);
            add3facumulate(radiance[Index].value, rad);
            
        }else {
            float inv_pdf = 1.0 / (sp.pdf * sp.rrpdf);
            float bsdf[3] = {0.0f};
            bsdfeval_device(sp, sp.wi, bsdf);
            float rad[3] = {0.0f};
            mul3fand3f(temp_rad[Index+1].value, bsdf, rad);
            mul3fand1f(rad, inv_pdf, rad);
            add3facumulate(radiance[Index].value, rad);
        }
    }
}

__global__ void copyValues(val3f *b, const val3f *a){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;
    copy3f(b[Index].value, a[Index].value);
}

__global__ void allGPUPdfSum(const SPoint *sps, const int *neighbors, const int *knn, float *pdfsum){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;
    int k = knn[Index];
    int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);
    for ( int i = 0 ; i < k ; i++){
        SPoint spi = sps[neighborhood[i]];
        if(spi.nidx > 0 && spi.rrpdf > 1e-7f){
            for( int j = 0 ; j < k ; j++){
                SPoint spo = sps[neighborhood[j]];
                float pdf = 0.0;
                pdf_device(spo, spi.wi, &pdf);
                pdfsum[neighboroffset+i] += pdf * spo.rrpdf;
            }
        }
    }
}

__global__ void initializeHashTable(GridEntry * cu_hash_table){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_num_clusters) return;

    cu_hash_table[Index].index = -1;
    cu_hash_table[Index].position[0] = 262144.0;
    cu_hash_table[Index].position[1] = 262144.0;
    cu_hash_table[Index].position[2] = 262144.0;
}

__global__ void allGPUPdfSumJitter(const SPoint *sps, const int *neighbors, const int *knn, float *pdfsum){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;
    int k = knn[Index];
    int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);
    for ( int i = 0 ; i < k ; i++){
        SPoint spi = sps[neighborhood[i]];
        if(spi.nidx > 0 && spi.rrpdf > 1e-7f){
            float temp_pdf = 0.0;
            for( int j = 0 ; j < k ; j++){
                SPoint spo = sps[neighborhood[j]];
                if (spi.groupIdx != spo.groupIdx) continue;
                float pdf = 0.0;
                pdf_device(spo, spi.wi, &pdf);
                temp_pdf += pdf * spo.rrpdf;
            }
            pdfsum[neighboroffset+i] = temp_pdf;
        }
    }
}

__global__ void allGPUDirectScatterRadiance(const SPoint *sps,  const LPoint *lps, const float2 *pdfmarginal, const int *neighbors, const int *knn, val3f *radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    int k = knn[Index];
    // int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);
    SPoint sp = sps[Index];
    
    float inv_lightpdfmarginal = 1.0 / pdfmarginal[Index].x;
    float inv_bsdfpdfmarginal = 1.0 / pdfmarginal[Index].y;
    for ( int i = 0 ; i < k ; i++){
        int index = neighborhood[i];
        SPoint spo = sps[index];
        float finalradiance[3] = {0.0};
        if( pdfmarginal[Index].x > 0.0){
            float bsdf[3] = {0.0f};
            float rad[3] = {0.0f};
            bsdfeval_device(spo, sp.wi_d, bsdf);
            mul3fand3f(lps[Index].L_directsample, bsdf, rad);
            mul3fand1f(rad, inv_lightpdfmarginal, rad);
            add3f(finalradiance, rad, finalradiance);
        }
        if( pdfmarginal[Index].y > 0.0){
            float bsdf[3] = {0.0f};
            float rad[3] = {0.0f};
            bsdfeval_device(spo, sp.wi, bsdf);
            mul3fand3f(lps[Index].L_bsdfsample, bsdf, rad);
            mul3fand1f(rad, inv_bsdfpdfmarginal, rad);
            add3f(finalradiance, rad, finalradiance);
        }

        atomicAdd(radiance[index].value, finalradiance[0]);
        atomicAdd(radiance[index].value+1, finalradiance[1]);
        atomicAdd(radiance[index].value+2, finalradiance[2]);
    }
}

__global__ void allGPUDirectScatterRadianceWithWeight(const SPoint *sps,  const LPoint *lps, const float2 *pdfmarginal, const int *neighbors, const int *knn, val3f *radiance, const float *weightsum, const float*minweight){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    int k = knn[Index];
    // int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);
    SPoint sp = sps[Index];
    
    /* if has continuous point on the path */
    float inv_lightpdfmarginal = 1.0 / pdfmarginal[Index].x;
    float inv_bsdfpdfmarginal = 1.0 / pdfmarginal[Index].y;
    float weight_s = weightsum[Index];
    float max_dist = minweight[Index];
    for ( int i = 0 ; i < k ; i++){
        int index = neighborhood[i];
        SPoint spo = sps[index];
        float finalradiance[3] = {0.0};
        if( pdfmarginal[Index].x > 0.0){
            float bsdf[3] = {0.0f};
            float rad[3] = {0.0f};
            bsdfeval_device(spo, sp.wi_d, bsdf);
            mul3fand3f(lps[Index].L_directsample, bsdf, rad);
            mul3fand1f(rad, inv_lightpdfmarginal, rad);
            add3f(finalradiance, rad, finalradiance);
        }

        if( pdfmarginal[Index].y > 0.0){
            float bsdf[3] = {0.0f};
            float rad[3] = {0.0f};
            bsdfeval_device(spo, sp.wi, bsdf);
            mul3fand3f(lps[Index].L_bsdfsample, bsdf, rad);
            mul3fand1f(rad, inv_bsdfpdfmarginal, rad);
            add3f(finalradiance, rad, finalradiance);
        }

        float dist = dev_distance(spo.pos, sp.pos);
        float weight =  (max_dist - 0.8 * dist) * weight_s;
        if(weight_s == 0.0) weight = 1.0;
        // if(max_dist < dist){
        //     printf("\n Index=%d,less dist_max = %.8f, dist =%.8f", Index, max_dist, dist);
        // }
        // printf("weight equals = %.8f \n", minweight[index]);
        atomicAdd(radiance[index].value, finalradiance[0] * weight);
        atomicAdd(radiance[index].value+1, finalradiance[1] * weight);
        atomicAdd(radiance[index].value+2, finalradiance[2] * weight);
    }
}

__global__ void allGPUPdfMarginal(const SPoint *sps, const int *neighbors, const int *knn, float *pdfmarginal){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    int k = knn[Index];
    // int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);

    SPoint spi = sps[Index];
    float pdfsum = 0.0;
    for ( int i = 0 ; i < k ; i++){
        SPoint spo = sps[neighborhood[i]];
        if(spo.nidx == Index) continue;
        float pdf = 0.0;
        pdf_device(spo, spi.wi, &pdf);
        pdfsum += pdf * spo.rrpdf;
    }
    pdfmarginal[Index] = pdfsum;
}

__global__ void allGPUPdfMarginalJitter(const SPoint *sps, const int *neighbors, const int *knn, float *pdfmarginal){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    int k = knn[Index];
    // int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);

    SPoint spi = sps[Index];
    float pdfsum = 0.0;
    for ( int i = 0 ; i < k ; i++){
        SPoint spo = sps[neighborhood[i]];
        if(spo.groupIdx != spi.groupIdx || spo.nidx == Index) continue;
        float pdf = 0.0;
        pdf_device(spo, spi.wi, &pdf);
        pdfsum += pdf * spo.rrpdf;
    }
    pdfmarginal[Index] = pdfsum;
}

__global__ void allGPUPdfMarginalAndWeight(const SPoint *sps, const int *neighbors, const int *knn, float *pdfmarginal, float *weightsum, float *count, float *minweight){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    int k = knn[Index];
    // int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);

    SPoint spi = sps[Index];
    float ws = 0.0;
    float max_dist = 0.0;
    float pdfm = 0.0;
    for ( int i = 0 ; i < k ; i++){
        int index = neighborhood[i];
        SPoint spo = sps[index];
        if(spo.nidx == Index) continue;
        float dist = dev_distance(spo.pos, spi.pos);
        max_dist = fmaxf(max_dist, dist);
    }

    for ( int i = 0 ; i < k ; i++){
        int index = neighborhood[i];
        SPoint spo = sps[index];
        if(spo.nidx == Index) continue;
        float pdf = 0.0;
        pdf_device(spo, spi.wi, &pdf);
        
        float dist = dev_distance(spo.pos, spi.pos);
        float w = max(max_dist - 0.8 * dist, 0.0);
        if (w < 0.0) {
            printf("\n w less than zero, %.8f, %.8f", max_dist, dist);
        }
        pdfm += pdf * w * spo.rrpdf;
        ws += w;
    }

    minweight[Index] = max_dist;
    if(ws != 0.0 && max_dist != 0.0){
        ws = k / ws;
    }else{
       ws = 0.0;
    }

    pdfmarginal[Index] = pdfm * ws;
    weightsum[Index] = ws;
}

__global__ void addtoSP(SPoint *cu_sps, const int *cu_clusters, const int *cu_clusters_offset){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_num_clusters) return;
    int pnum = getCellSize(cu_clusters_offset, Index, d_num_clusters, d_num_clusters_point);
    int offset = cu_clusters_offset[Index];
    for (int i = 0 ; i < pnum ; i++){
        int idx = cu_clusters[offset+i];
        cu_sps[idx].groupIdx = Index;
    }
}

__global__ void allGPUDirectPdfMarginalAndWeight(const SPoint *sps, const LPoint * lps, const int *neighbors, const int *knn, float2 *pdfmarginal, float * weightsum, float * count, float *minweight){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    int k = knn[Index];
    // int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);

    SPoint spi = sps[Index];
    float pdfm = 0.0;
    float pdfl = 0.0;
    float ws = 0.0;
    float max_dist = 0.0;
    for ( int i = 0 ; i < k ; i++){
        int index = neighborhood[i];
        SPoint spo = sps[index];
        float dist = dev_distance(spo.pos, spi.pos);
        max_dist = max(max_dist, dist);
    }


    for ( int i = 0 ; i < k ; i++){
        int index = neighborhood[i];
        SPoint spo = sps[index];
        float pdf_bsdf = 0.0;
        pdf_device(spo, spi.wi, &pdf_bsdf);

        float dist = dev_distance(spo.pos, spi.pos);
        
        float w = max_dist - 0.8 * dist;
        ws += w;
        pdfm += pdf_bsdf * w;
        pdfl += lps[Index].lightpdf * w;
    }
    minweight[Index] = max_dist;
    if(ws != 0.0 && max_dist != 0.0){
        ws = k / ws;
    }else{
        ws = 0.0;
    }
    pdfmarginal[Index].x = pdfl * ws;
    pdfmarginal[Index].y = pdfm * ws;
    weightsum[Index] = ws;
}

__global__ void allGPUDirectPdfMarginal(const SPoint *sps, const LPoint * lps, const int *neighbors, const int *knn, float2 *pdfmarginal){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    int k = knn[Index];
    // int neighboroffset = d_k * Index;
    int *neighborhood = (int *) ((char *)neighbors + Index * d_pitch_n);

    SPoint spi = sps[Index];
    for ( int i = 0 ; i < k ; i++){
        SPoint spo = sps[neighborhood[i]];
        float pdf_bsdf = 0.0;
        pdf_device(spo, spi.wi, &pdf_bsdf);
        pdfmarginal[Index].x += lps[Index].lightpdf;
        pdfmarginal[Index].y += pdf_bsdf;
    }
}

__global__ void ClusterDirectPdfMarginal(const SPoint *sps, const LPoint * lps, const int *cu_clusters, const int *cu_cluster_offset, float2 *pdfmarginal){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    SPoint spi = sps[Index];
    int cluster_idx = spi.groupIdx;
    if (cluster_idx != -1){
        int k = getCellSize(cu_cluster_offset, cluster_idx, d_num_clusters, d_num_clusters_point);
        int my_cluster_offset = cu_cluster_offset[cluster_idx];

        for ( int i = 0 ; i < k ; i++){
            int neighbor_idx = cu_clusters[my_cluster_offset+i];
            SPoint spo = sps[neighbor_idx];
            float light_bsdf_pdf = 0.0;
            float bsdf_bsdf_pdf = 0.0;
            pdf_device(spo, spi.wi, &bsdf_bsdf_pdf);
            pdf_device(spo, spi.wi_d, &light_bsdf_pdf);
            float cosh1 = dot(spi.wi, spo.shN);
            float cosh2 = dot(spi.wi_d, spo.shN);
            if(cosh2 > 0.0)
                pdfmarginal[Index].x += lps[Index].lightpdf;
            if(cosh1 > 0.0) 
                pdfmarginal[Index].y += lps[Index].bsdfpdf;
            pdfmarginal[Index].x += light_bsdf_pdf;
            pdfmarginal[Index].y += bsdf_bsdf_pdf; // bsdf_light_pdf , bsdf_bsdf_pdf
            // if(Index == 2978795){
            //     printf(" Direct light light_bsdf_pdf = %.4f, bsdf_bsdf_pdf = %.4f, lightpdf = %.4f \n", light_bsdf_pdf, bsdf_bsdf_pdf, lps[Index].lightpdf);
            // }
        }
    }else{
        float light_bsdf_pdf = 0.0;
        float bsdf_bsdf_pdf = 0.0;
        pdf_device(spi, spi.wi, &bsdf_bsdf_pdf);
        pdf_device(spi, spi.wi_d, &light_bsdf_pdf);
        pdfmarginal[Index].x = lps[Index].lightpdf + light_bsdf_pdf;
        pdfmarginal[Index].y = lps[Index].bsdfpdf + bsdf_bsdf_pdf;
    }
}


__global__ void ClusterDirectRadiance(const SPoint *sps,  const LPoint *lps, const float2 *pdfmarginal, const int *cu_clusters, const int *cu_cluster_offset, val3f *radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if (Index >= d_sizeofsp) return;

    SPoint sp = sps[Index];
    int cluster_idx = sp.groupIdx;
    if (cluster_idx != -1){
        int k = getCellSize(cu_cluster_offset, cluster_idx, d_num_clusters, d_num_clusters_point);
        int my_offset = cu_cluster_offset[cluster_idx];

        float inv_lightpdfmarginal = 1.0 / pdfmarginal[Index].x;
        float inv_bsdfpdfmarginal = 1.0 / pdfmarginal[Index].y;
        // float inv_pdfmarginal = 1.0 / (pdfmarginal[Index].x + pdfmarginal[Index].y);

        for ( int i = 0 ; i < k ; i++){
            int index = cu_clusters[my_offset + i];
            SPoint spo = sps[index];
            float finalradiance[3] = {0.0};
            if( pdfmarginal[Index].x > 0.0){
                float bsdf[3] = {0.0f};
                float rad[3] = {0.0f};
                bool t = false;
                // if(index == 2978795 && Index == 1903118) t = true;
                bsdfeval_device(spo, sp.wi_d, bsdf, t);
                // float light_bsdf_pdf = 0.0;
                // pdf_device(spo, sp.wi_d, &light_bsdf_pdf);
                // float weight_light_sample = light_bsdf_pdf * inv_lightpdfmarginal;
                mul3fand3f(lps[Index].L_directsample, bsdf, rad);
                mul3fand1f(rad, inv_lightpdfmarginal, rad);
                // mul3fand1f(rad, inv_pdfmarginal, rad);
                add3f(finalradiance, rad, finalradiance);
                // if (index == 2978795){
                //     printf("%d, direct sample: bsdf = [%.4f, %.4f, %.4f], lps = [%.4f, %.4f, %.4f], inv_pdf = %.4f\n finalradiance = [%.4f, %.4f, %.4f] \n", Index,
                //     bsdf[0], bsdf[1], bsdf[2], lps[Index].L_directsample[0], lps[Index].L_directsample[1], lps[Index].L_directsample[2], inv_lightpdfmarginal, finalradiance[0], finalradiance[1], finalradiance[2]);
                // }
            }
            if( pdfmarginal[Index].y > 0.0){
                float bsdf[3] = {0.0f};
                float rad[3] = {0.0f};
                bsdfeval_device(spo, sp.wi, bsdf);
                mul3fand3f(lps[Index].L_bsdfsample, bsdf, rad);
                mul3fand1f(rad, inv_bsdfpdfmarginal, rad);
                // mul3fand1f(rad, inv_pdfmarginal, rad);
                add3f(finalradiance, rad, finalradiance);
                // if (index == 2978795){
                //     printf("%d, bsdf sample: bsdf = [%.4f, %.4f, %.4f], lps = [%.4f, %.4f, %.4f], invpdf = %.4f\n", Index,
                //     bsdf[0], bsdf[1], bsdf[2], lps[Index].L_bsdfsample[0], lps[Index].L_bsdfsample[1], lps[Index].L_bsdfsample[2], inv_bsdfpdfmarginal);
                // }
            }

            atomicAdd(radiance[index].value, finalradiance[0]);
            atomicAdd(radiance[index].value+1, finalradiance[1]);
            atomicAdd(radiance[index].value+2, finalradiance[2]);
            // if (index == 2978795){
            //     printf("final radiance = [%.4f, %.4f, %.4f]\n",
            //     finalradiance[0], finalradiance[1], finalradiance[2]);
            // }
        }
    }else{
        atomicAdd(radiance[Index].value, sp.eLd[0]);
        atomicAdd(radiance[Index].value+1, sp.eLd[1]);
        atomicAdd(radiance[Index].value+2, sp.eLd[2]);
    }
    
}

/* pdf of sampling wi at given shading point */
__global__ void pdfeval(const SPoint *sps, float *pdf){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    SPoint sp = sps[Index];
    float dotWiShN = dot(sp.wi, sp.shN);
    float dotWigeoN = dot(sp.wi, sp.geoN);
    if( dotWigeoN  * dotWiShN <= 0.0) return;
    float dotWoGeoN = dot(sp.wo, sp.geoN);
    float dotWoShN = dot(sp.wo, sp.shN);
    if( dotWiShN <= 0.0 || dotWoShN <= 0.0) return;

    float wh[3];
    add3f(sp.wi, sp.wo, wh);
    normalize(wh);
    float dotWhShN = dot(wh, sp.shN);
    float dotWiH = dot(sp.wi, wh);
    float dotWoH = dot(sp.wo, wh);
    float inv_dWhWi = 1.0 / (4.0 * dotWiH);
    float prob = distrpdf(dotWoShN, dotWoH, dotWhShN, sp.roughness);
    float specular = prob * inv_dWhWi * 0.5;
    float diffuse = 0.5 * dotWiShN * M_1_PI;
    *pdf = specular + diffuse;
}

/* pdf evaluation of scattering method */
__global__ void pdfWo(SPoint *sps, int *shadingIdx, float *pdfmargin){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    SPoint sp = sps[shadingIdx[Index]];
    float pdf = 0.0;
    pdf_device(sp, sps[Index].wi, &pdf); 
    pdfmargin[Index] += pdf;
}

__global__ void pdfWoWi(SPoint *sps, int *shadingIdx, int * neighborsIdx, float *pdfsum){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    SPoint sp = sps[shadingIdx[Index]];
    int nidx = neighborsIdx[Index];
    float pdf = 0.0;
    pdf_device(sp, sps[nidx].wi, &pdf);
    pdfsum[Index] += (pdf);
}

__global__ void bsdfeval(SPoint *sps, val3f * result){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    SPoint sp = sps[Index];
    float dotWiShN = dot(sp.wi, sp.shN);
    float dotWigeoN = dot(sp.wi, sp.geoN);
    if( dotWigeoN  * dotWiShN <= 0.0) return;
    float dotWoGeoN = dot(sp.wo, sp.geoN);
    float dotWoShN = dot(sp.wo, sp.shN);
    if( dotWiShN <= 0.0 || dotWoShN <= 0.0) return;

    /* compute specular part */
    float wh[3];
    add3f(sp.wi, sp.wo, wh);
    normalize(wh);
    float dotWhShN = dot(wh, sp.shN);
    float D = distreval(dotWhShN, sp.roughness);

    float dotWoH = dot(sp.wo, wh);
    float dotWiH = dot(sp.wi, wh);
    float F = fresnelDielectricExt(dotWoH, 1.5);
    float G = smithG1(dotWoH, dotWoShN, sp.roughness) * smithG1(dotWiH, dotWiShN, sp.roughness);

    float specularconst = F * G * D / (4.0f * dotWoShN);
    float specular[3];
    mul3fand1f(sp.specular, specularconst, specular);
    /*compute diffuse part*/
    float diffuseconst = M_1_PI * dotWiShN;
    float diffuse[3];
    mul3fand1f(sp.diffuse, diffuseconst, diffuse);
    add3f(diffuse, specular, result[Index].value);
}


__global__ void computeRadiance(const SPoint *sps, const val3f *bsdf, const float *pdf, val3f *radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    int idx = sps[Index].nidx;
    if (idx > 0 && pdf[Index] > 0.0){
        float inv_pdf = 1.0 / pdf[Index];
        mul3fand3f(sps[Index+1].eLi, bsdf[Index].value, radiance[Index].value);
        mul3fand1f(radiance[Index].value, inv_pdf, radiance[Index].value);
    }else zero3f(radiance[Index].value); 
}

__global__ void clampDirectCluster(const SPoint *sps, const LPoint *lps, val3f *cu_cluster_out, val3f * cu_cluster_in, const val3f *radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    SPoint sp = sps[Index];
    int idx = sp.groupIdx;
    
    if(idx == -1) return;

    // if(Index == 2978795) {
    //     printf(" Index = 2978795 , group idx = %d \n ============ \n", idx);
    //     d_debug_cluster_idx = idx;
    // }

    atomicAdd(cu_cluster_in[idx].value, lps[Index].L_directsample[0]);
    atomicAdd(cu_cluster_in[idx].value+1, lps[Index].L_directsample[1]);
    atomicAdd(cu_cluster_in[idx].value+2, lps[Index].L_directsample[2]);

    atomicAdd(cu_cluster_in[idx].value, lps[Index].L_bsdfsample[0]);
    atomicAdd(cu_cluster_in[idx].value+1, lps[Index].L_bsdfsample[1]);
    atomicAdd(cu_cluster_in[idx].value+2, lps[Index].L_bsdfsample[2]);

    atomicAdd(cu_cluster_out[idx].value, radiance[Index].value[0]);
    atomicAdd(cu_cluster_out[idx].value+1, radiance[Index].value[1]);
    atomicAdd(cu_cluster_out[idx].value+2, radiance[Index].value[2]);
}

__global__ void clampCluster(const SPoint *sps, val3f *cu_cluster_out, val3f * cu_cluster_in, const val3f *radiance, const val3f *temp_radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    SPoint sp = sps[Index];
    int idx = sp.groupIdx;
    
    if(idx == -1) return;

    if (sp.nidx > 0){
        atomicAdd(cu_cluster_in[idx].value, temp_radiance[Index+1].value[0]);
        atomicAdd(cu_cluster_in[idx].value+1, temp_radiance[Index+1].value[1]);
        atomicAdd(cu_cluster_in[idx].value+2, temp_radiance[Index+1].value[2]);
    }
    atomicAdd(cu_cluster_out[idx].value, radiance[Index].value[0]);
    atomicAdd(cu_cluster_out[idx].value+1, radiance[Index].value[1]);
    atomicAdd(cu_cluster_out[idx].value+2, radiance[Index].value[2]);
}

__global__ void computeRatio(const SPoint *sps, const val3f *cu_cluster_out, const val3f * cu_cluster_in, val3f * cu_ratio){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_num_clusters) return;

    val3f in_rad = cu_cluster_in[Index];
    val3f out_rad = cu_cluster_out[Index];
    cu_ratio[Index].value[0] = 1.0f;
    cu_ratio[Index].value[1] = 1.0f;
    cu_ratio[Index].value[2] = 1.0f;

    if(in_rad.value[0] < out_rad.value[0]){
        cu_ratio[Index].value[0] = in_rad.value[0] / out_rad.value[0];
        
    }
    
    if(in_rad.value[1] < out_rad.value[1]){
        cu_ratio[Index].value[1] = in_rad.value[1] / out_rad.value[1];
    }

    // if (Index == d_debug_cluster_idx) printf("\n red = %.3f, red out = %.3f", in_rad.value[0], out_rad.value[0]);
    // if (Index == d_debug_cluster_idx) printf("\n green = %.3f, green out = %.3f", in_rad.value[1], out_rad.value[1]);
    // if (Index == d_debug_cluster_idx) printf("\n blue = %.3f, blue out = %.3f", in_rad.value[2], out_rad.value[2]);

    if(in_rad.value[2] < out_rad.value[2]){
        cu_ratio[Index].value[2] = in_rad.value[2] / out_rad.value[2];    
    }
}

__global__ void updateComputeCluster(const SPoint *sps, val3f *radiance, const val3f *cu_cluster_ratio){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;
    SPoint sp = sps[Index];
    int idx = sp.groupIdx;
    if (idx != -1){
        mul3fand3f(radiance[Index].value, cu_cluster_ratio[idx].value, radiance[Index].value);
    }
    
}

__global__ void updateRadiance(const SPoint *sps, val3f * cu_tempRadiance, const val3f * cu_radiance,  int j){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    if(j == 0){
        copy3f(cu_tempRadiance[Index].value, sps[Index].eLi);
    }else{
        add3f(cu_radiance[Index].value, sps[Index].eLd, cu_tempRadiance[Index].value);
    }
}

__global__ void updateWithOptDirectRadiance(const val3f * cu_direct_radiance, val3f * cu_tempRadiance, const val3f * cu_radiance){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;
    add3f(cu_direct_radiance[Index].value, cu_radiance[Index].value, cu_tempRadiance[Index].value);
    // if(Index == 2978795) 
    //     printf("radiance = [%.4f, %.4f, %.4f]", cu_radiance[Index].value[0], cu_radiance[Index].value[1], cu_radiance[Index].value[2]);
}

__global__ void computeScatterRadiance(const SPoint *sps, const int *neIdx,const float *pdf_marginal, const val3f *radiance, val3f *ret){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    int comingLightIdx = sps[Index].nidx;
    if (comingLightIdx > 0 &&  pdf_marginal[Index] > 0.0f){
        int index = neIdx[Index];
        SPoint sp = sps[Index];
        float bsdf[3] = {0.0f};
        float inv_pdfm = 1.0 / pdf_marginal[Index];
        float rad[3] = {0.0f};
        bsdfeval_device(sps[index], sp.wi, bsdf);
        mul3fand3f(radiance[comingLightIdx].value, bsdf, rad);
        mul3fand1f(rad, inv_pdfm, rad);
        atomicAdd(ret[index].value, rad[0]);
        atomicAdd(ret[index].value+1, rad[1]);
        atomicAdd(ret[index].value+2, rad[2]);
    }
}

__global__ void clampRadiance(val3f *ret){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    // if(Index == 2978795)
    //     printf("radiance = [%.4f, %.4f, %.4f]\n", ret[Index].value[0], ret[Index].value[1], ret[Index].value[2]);

    if(isnan(ret[Index].value[0]) || isnan(ret[Index].value[1]), isnan(ret[Index].value[2]))
        printf("==========IS NAN %d, [%.4f, %.4f, %.4f] ===========", Index, ret[Index].value[0], ret[Index].value[1], ret[Index].value[2]);

    // if ( abs(ret[Index].value[0] - 318) < 1 && abs(ret[Index].value[1] - 271 < 1) && abs(ret[Index].value[1] - 196) < 1 )
    //     printf("Index = %d, radiance = [%.4f, %.4f, %.4f]\n", Index, ret[Index].value[0], ret[Index].value[1], ret[Index].value[2]);

    // for (int i = 0 ; i < 3 ; i++){
    //     if(ret[Index].value[i] > 10.0)
    //         ret[Index].value[i] = 10.0;
    // }
}


__global__ void setValue(float *array, float value){
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index >= d_sizeofsp) return;

    array[Index] = value;
}

/* Basic wraped function call for cpp */
void CUDAErrorLog(int error, std::string ctr){
    std::cout<<"["<<ctr<<"]: "<<cudaGetErrorName((cudaError_t)error)<<std::endl;
}

void CUDA_ERROR_CHECK(cudaError_t error){
    if(error != 0)
        std::cout<<"[Error!]: "<<cudaGetErrorName((cudaError_t)error)<<std::endl;
}

bool CUDAmalloc(void ** A, size_t size_in_byte){
    int status = cudaMalloc(A, size_in_byte);
    if(status == 0) return true;
    CUDAErrorLog(status, "malloc ");
    return false;
}

bool CUDAmallocPitch(void **addr, size_t *pitch, size_t height_size_in_byte, size_t width){
    int status = cudaMallocPitch(addr, pitch, height_size_in_byte, width);
    if(status == 0) return true;
    CUDAErrorLog(status, "mallocPitch ");
    return false;
}

bool CUDAcpyH2D(void *device, void *host, size_t size_in_byte){
    int status = cudaMemcpy(device, host, size_in_byte, cudaMemcpyHostToDevice);
    if(status == 0) return true;
    CUDAErrorLog(status, "copy host to device");
    return false;
}

bool CUDAcpyD2H(void *device, void *host, size_t size_in_byte){
    int status = cudaMemcpy(host, device, size_in_byte, cudaMemcpyDeviceToHost);
    if(status == 0) return true;
    CUDAErrorLog(status, "copy device to host");
    return false;
}

bool CUDAmemD2D(void *dst, void *src, size_t size_in_byte){
    int status = cudaMemcpy(dst, src, size_in_byte, cudaMemcpyDeviceToDevice);
    if(status == 0) return true;
    CUDAErrorLog(status, "copy device to device");
    return false;
}

void CUDAdelete(void *device){
   cudaFree(device);
}


void CUDAcheckMemory(int i){
    size_t free_t, total_t;
    cudaMemGetInfo(&free_t, &total_t);
    float free_m, total_m, used_m;
    free_m = free_t / 1048576.0;
    total_m = total_t / 1048576.0;
    used_m = total_m - free_m;
    printf ( "\t [%d]  mem free %d .... %f MB mem total %d....%f MB mem used %f MB\n",i, free_t,free_m,total_t,total_m,used_m);
}

void plotRange(float *cu_float, uint32_t size){
    std::vector<float> values(size);
    CUDAcpyD2H(cu_float, values.data(), size) * sizeof(float);
    printf("\n \t [max]=%.5f, [min]=%.5f\n", *std::max_element(values.begin(), values.end()),*std::min_element(values.begin(), values.end()));
}

void computeScatterAllOnGPURecord(const SPoint *cu_sps, const int *cu_neighbors, const int *cu_knn, int sizeofsp, int iterations, int k, ResultSpace &ret, const size_t pitch_n){
    std::cout<<"starting computing scattering radiance!"<<std::endl;
    size_t pdfsum_size_in_byte = sizeofsp * sizeof(float);
    size_t size_radiance_in_byte = sizeofsp * sizeof(val3f);
    float *cu_pdfmarginal; //*cu_weightsum;
    val3f *cu_tempRadiance, *cu_radiance;    
    cudaMalloc((void **)&cu_pdfmarginal, pdfsum_size_in_byte);
    
    /*copy constant to symbol*/
    printf("the max k I use: %d\n", k);
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_pitch_n, &pitch_n, sizeof(size_t));

    /*compute marginal pdf sum*/
    int blocknum = sizeofsp / THREADDIM + 1;
    cudaMemset((void *)cu_pdfmarginal, 0, pdfsum_size_in_byte);
    allGPUPdfMarginal<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors,cu_knn, cu_pdfmarginal);
    cudaDeviceSynchronize();
    
    /*Allocate and initialize*/
    CUDAmalloc((void **)&cu_radiance, size_radiance_in_byte);
    CUDAmalloc((void **)&cu_tempRadiance, size_radiance_in_byte);
    cudaMemset((void *)cu_radiance, 0, size_radiance_in_byte);
    cudaMemset((void *)cu_tempRadiance, 0, size_radiance_in_byte);


    int i=0;
    updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
    for (; i < iterations ; i++){
        cudaMemset((void *)cu_radiance, 0, size_radiance_in_byte);
        // cudaMemset((void *)cu_weightsum, 0, pdfsum_size_in_byte);
        allGPUScatterRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfmarginal, cu_neighbors, cu_knn, cu_tempRadiance, cu_radiance);
        // plotRange(cu_weightsum, sizeofsp);
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.blur_results[i], size_radiance_in_byte);
        cudaDeviceSynchronize();
        updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
        cudaDeviceSynchronize();
        cudaMemset((void *)cu_radiance, 0, size_radiance_in_byte);
        lastRun<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance);
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.mc_results[i], size_radiance_in_byte);
        cudaDeviceSynchronize();
    }
    cudaFree(cu_radiance);
    cudaFree(cu_tempRadiance);
    cudaFree(cu_pdfmarginal);
    // cudaFree(cu_weightsum);
}

void computeDirectScatterAllOnGPU(const SPoint *cu_sps,const LPoint *cu_lps, const int *cu_neighbors, const int *cu_dknn, int sizeofsp, int k, val3f * cu_direct_radiance, const size_t pitch_n){
    std::cout<<"starting computing scattering radiance!"<<std::endl;
    int blocknum = sizeofsp / THREADDIM + 1;
    size_t directpdfmarginal_size_in_byte = sizeofsp * sizeof(float2);
    size_t radiance_size_in_byte = sizeofsp * sizeof(val3f);
    float2 *cu_directpdfmarginal;
    
    /*copy constant to symbol*/
    printf("k I use in direct: %d\n", k);
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_pitch_n, &pitch_n, sizeof(size_t));

    /*compute direct marginal pdf sum*/
    CUDAmalloc((void **)&cu_directpdfmarginal, directpdfmarginal_size_in_byte);
    cudaMemset((void *)cu_directpdfmarginal, 0, directpdfmarginal_size_in_byte);
    allGPUDirectPdfMarginal<<<blocknum, THREADDIM>>>(cu_sps, cu_lps, cu_neighbors, cu_dknn, cu_directpdfmarginal);
    cudaDeviceSynchronize();

    cudaMemset((void *)cu_direct_radiance, 0, radiance_size_in_byte);
    allGPUDirectScatterRadiance<<<blocknum, THREADDIM>>>(cu_sps,  cu_lps, cu_directpdfmarginal, cu_neighbors, cu_dknn, cu_direct_radiance);
    cudaDeviceSynchronize();
    addEmitterToDirectLight<<<blocknum, THREADDIM>>>(cu_lps, cu_direct_radiance);
    cudaDeviceSynchronize();

    CUDAdelete(cu_directpdfmarginal);
}



void ClusterDirect(const SPoint *cu_sps,const LPoint *cu_lps, const int *cu_clusters, const int *cu_clusters_offset, int sizeofsp, int num_of_clusters, val3f * cu_direct_radiance){
    std::cout<<"starting computing direct radiance!"<<std::endl;
    int blocknum = sizeofsp / THREADDIM + 1;
    size_t directpdfmarginal_size_in_byte = sizeofsp * sizeof(float2);
    size_t radiance_size_in_byte = sizeofsp * sizeof(val3f);
    float2 *cu_directpdfmarginal;
    
    /*copy constant to symbol*/
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));

    /*compute direct marginal pdf sum*/
    CUDAmalloc((void **)&cu_directpdfmarginal, directpdfmarginal_size_in_byte);
    cudaMemset((void *)cu_directpdfmarginal, 0, directpdfmarginal_size_in_byte);
    ClusterDirectPdfMarginal<<<blocknum, THREADDIM>>>(cu_sps, cu_lps, cu_clusters, cu_clusters_offset, cu_directpdfmarginal);
    cudaDeviceSynchronize();

    cudaMemset((void *)cu_direct_radiance, 0, radiance_size_in_byte);
    ClusterDirectRadiance<<<blocknum, THREADDIM>>>(cu_sps,  cu_lps, cu_directpdfmarginal, cu_clusters, cu_clusters_offset, cu_direct_radiance);
    cudaDeviceSynchronize();
    
    val3f *cu_cluster_out, *cu_cluster_in, *cu_cluster_ratio; 
    CUDAmalloc((void **)&cu_cluster_out, sizeof(val3f) * num_of_clusters);
    CUDAmalloc((void **)&cu_cluster_in, sizeof(val3f) * num_of_clusters);
    CUDAmalloc((void **)&cu_cluster_ratio, sizeof(val3f) * num_of_clusters);
    cudaMemset((void *)cu_cluster_out, 0, sizeof(val3f) * num_of_clusters);
    cudaMemset((void *)cu_cluster_in, 0, sizeof(val3f) * num_of_clusters);
    cudaMemset((void *)cu_cluster_ratio, 0, sizeof(val3f) * num_of_clusters);

    int blocknum2 = num_of_clusters / THREADDIM + 1;

    clampDirectCluster<<<blocknum, THREADDIM>>>(cu_sps, cu_lps, cu_cluster_out, cu_cluster_in, cu_direct_radiance);
    cudaDeviceSynchronize();
    computeRatio<<<blocknum2, THREADDIM>>>(cu_sps, cu_cluster_out, cu_cluster_in, cu_cluster_ratio);
    cudaDeviceSynchronize();
    updateComputeCluster<<<blocknum, THREADDIM>>>(cu_sps, cu_direct_radiance, cu_cluster_ratio);
    cudaDeviceSynchronize();
    addEmitterToDirectLight<<<blocknum, THREADDIM>>>(cu_lps, cu_direct_radiance);
    cudaDeviceSynchronize();

    CUDAdelete(cu_directpdfmarginal);
    CUDAdelete(cu_cluster_out);
    CUDAdelete(cu_cluster_in);
    CUDAdelete(cu_cluster_ratio);
}


void computeDirectScatterAllOnGPUWithWeight(const SPoint *cu_sps,const LPoint *cu_lps, const int *cu_neighbors, const int *cu_dknn, int sizeofsp, int k, val3f * cu_direct_radiance, const size_t pitch_n){
    std::cout<<"starting computing scattering radiance!"<<std::endl;
    int blocknum = sizeofsp / THREADDIM + 1;
    size_t directpdfmarginal_size_in_byte = sizeofsp * sizeof(float2);
    size_t radiance_size_in_byte = sizeofsp * sizeof(val3f);
    float2 *cu_directpdfmarginal;
    float *cu_directweightsum;
    float *cu_count;
    float *cu_minweight;
    
    /*copy constant to symbol*/
    printf("k I use in direct: %d\n", k);
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_pitch_n, &pitch_n, sizeof(size_t));

    /*compute direct marginal pdf sum*/
    CUDAmalloc((void **)&cu_directpdfmarginal, directpdfmarginal_size_in_byte);
    CUDAmalloc((void **)&cu_directweightsum, sizeofsp * sizeof(float));
    CUDAmalloc((void **)&cu_count, sizeofsp * sizeof(float));
    CUDAmalloc((void **)&cu_minweight, sizeofsp * sizeof(float));


    cudaMemset((void *)cu_directpdfmarginal, 0, directpdfmarginal_size_in_byte);
    cudaMemset((void *)cu_count, 0, sizeofsp * sizeof(float));
    cudaMemset((void *)cu_directweightsum, 0, sizeofsp * sizeof(float));
    setValue<<<blocknum, THREADDIM>>>(cu_minweight, 0.0);

    allGPUDirectPdfMarginalAndWeight<<<blocknum, THREADDIM>>>(cu_sps, cu_lps, cu_neighbors, cu_dknn, cu_directpdfmarginal, cu_directweightsum, cu_count, cu_minweight);
    cudaFree(cu_count);

    cudaMemset((void *)cu_direct_radiance, 0, radiance_size_in_byte);
    allGPUDirectScatterRadianceWithWeight<<<blocknum, THREADDIM>>>(cu_sps,  cu_lps, cu_directpdfmarginal, cu_neighbors, cu_dknn, cu_direct_radiance, cu_directweightsum, cu_minweight);
    cudaDeviceSynchronize();
    addEmitterToDirectLight<<<blocknum, THREADDIM>>>(cu_lps, cu_direct_radiance);
    cudaDeviceSynchronize();

    CUDAdelete(cu_directpdfmarginal);
    CUDAdelete(cu_directweightsum);
    CUDAdelete(cu_minweight);
}

void computeScatterAllOnGPURecordWithDirectOpt(const SPoint *cu_sps,const int *cu_neighbors, const int *cu_knn, 
    int sizeofsp, int iterations, int k, val3f * cu_direct_radiance, ResultSpace ret, const size_t pitch_n){
    std::cout<<"starting computing scattering radiance!"<<std::endl;
    size_t pdfsum_size_in_byte = sizeofsp * sizeof(float);
    size_t radiance_size_in_byte = sizeofsp * sizeof(val3f);
    float *cu_pdfmarginal, * cu_weightsum;
    val3f *cu_tempRadiance, *cu_radiance;    
    
    
    /*copy constant to symbol*/
    printf("the maxk k I use for indirect: %d", k);
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_pitch_n, &pitch_n, sizeof(size_t));

    /*compute marginal pdf sum*/
    int blocknum = sizeofsp / THREADDIM + 1;
    cudaMalloc((void **)&cu_pdfmarginal, pdfsum_size_in_byte);
    cudaMalloc((void **)&cu_weightsum, pdfsum_size_in_byte);
    cudaMemset((void *)cu_pdfmarginal, 0, pdfsum_size_in_byte);
    cudaMemset((void *)cu_weightsum, 0, pdfsum_size_in_byte);
    allGPUPdfMarginal<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors, cu_knn, cu_pdfmarginal);
    cudaDeviceSynchronize();
    
    /*Allocate and initialize*/
    CUDAmalloc((void **)&cu_radiance, radiance_size_in_byte);
    CUDAmalloc((void **)&cu_tempRadiance, radiance_size_in_byte);
    cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
    cudaMemset((void *)cu_tempRadiance, 0, radiance_size_in_byte);
    int i=0;
    updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
    for (; i < iterations ; i++){
        cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
        allGPUScatterRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfmarginal, cu_neighbors, cu_knn, cu_tempRadiance, cu_radiance);
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.blur_results[i], radiance_size_in_byte);
        cudaDeviceSynchronize();
        updateWithOptDirectRadiance<<<blocknum, THREADDIM>>>(cu_direct_radiance, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
        lastRun<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.mc_results[i], radiance_size_in_byte);
        cudaDeviceSynchronize();
    }
    cudaFree(cu_radiance);
    cudaFree(cu_tempRadiance);
    cudaFree(cu_pdfmarginal);
    cudaFree(cu_weightsum);
}


void computeScatterAllOnGPUWithDirectOpt(const SPoint *cu_sps,const int *cu_neighbors, const int *cu_knn, const int *cu_dknn,
    int sizeofsp, int iterations, int k, int dk, val3f * cu_direct_radiance, ResultSpace ret, const size_t pitch_n, bool jitter){
    std::cout<<"starting computing scattering radiance!"<<std::endl;
    size_t pdfsum_size_in_byte = sizeofsp * sizeof(float);
    size_t radiance_size_in_byte = sizeofsp * sizeof(val3f);
    float *cu_pdfmarginal;// * cu_weightsum;
    val3f *cu_tempRadiance, *cu_radiance;    
    
    
    /*copy constant to symbol*/
    printf("the maxk k I use for indirect: %d", k);
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_pitch_n, &pitch_n, sizeof(size_t));

    /*compute marginal pdf sum*/
    int blocknum = sizeofsp / THREADDIM + 1;
    cudaMalloc((void **)&cu_pdfmarginal, pdfsum_size_in_byte);
    cudaMemset((void *)cu_pdfmarginal, 0, pdfsum_size_in_byte);
    allGPUPdfMarginal<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors, cu_knn, cu_pdfmarginal);
    cudaDeviceSynchronize();
    
    /*Allocate and initialize*/
    CUDAmalloc((void **)&cu_radiance, radiance_size_in_byte);
    CUDAmalloc((void **)&cu_tempRadiance, radiance_size_in_byte);
    cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
    cudaMemset((void *)cu_tempRadiance, 0, radiance_size_in_byte);
    int i=0;
    updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
    for (; i < iterations ; i++){
        cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
        allGPUScatterRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfmarginal, cu_neighbors, cu_knn, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        if (i == iterations-1) CUDAcpyD2H((void *)cu_radiance, (void *)ret.blur_results[0], radiance_size_in_byte);
        updateWithOptDirectRadiance<<<blocknum, THREADDIM>>>(cu_direct_radiance, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        if (i == iterations-1){
            cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
            if(jitter){
                cudaMemset((void *)cu_pdfmarginal, 0, pdfsum_size_in_byte);
                if (dk <= k){
                    allGPUPdfMarginalJitter<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors, cu_knn, cu_pdfmarginal);
                    lastRunJitter<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfmarginal, cu_neighbors, cu_knn, cu_tempRadiance, cu_radiance);
                }else{
                    cudaMemcpyToSymbol(d_k, &dk, sizeof(int));
                    allGPUPdfMarginalJitter<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors, cu_dknn, cu_pdfmarginal);
                    lastRunJitter<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfmarginal, cu_neighbors, cu_dknn, cu_tempRadiance, cu_radiance);
                } 
            }else{
                lastRun<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance);
            }
            cudaDeviceSynchronize();
            clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
            cudaDeviceSynchronize();
            CUDAcpyD2H((void *)cu_radiance, (void *)ret.mc_results[0], radiance_size_in_byte);
        } 
    }
    cudaFree(cu_radiance);
    cudaFree(cu_tempRadiance);
    cudaFree(cu_pdfmarginal);
}

void computeScatterAllOnGPURecordWithDirectOptandWeight(const SPoint *cu_sps,const int *cu_neighbors, 
    const int *cu_knn, int sizeofsp, int iterations, int k, val3f * cu_direct_radiance, ResultSpace ret, const size_t pitch_n){
    std::cout<<"starting computing scattering radiance!"<<std::endl;
    size_t pdfsum_size_in_byte = sizeofsp * sizeof(float);
    size_t radiance_size_in_byte = sizeofsp * sizeof(val3f);
    float *cu_pdfmarginal, *cu_weightsum, *cu_minweights, *cu_count;
    val3f *cu_tempRadiance, *cu_radiance;    
    
    /*copy constant to symbol*/
    printf("the maxk k I use for indirect: %d", k);
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_pitch_n, &pitch_n, sizeof(size_t));

    /*compute marginal pdf sum*/
    int blocknum = sizeofsp / THREADDIM + 1;
    cudaMalloc((void **)&cu_pdfmarginal, pdfsum_size_in_byte);
    cudaMalloc((void **)&cu_weightsum, pdfsum_size_in_byte);
    cudaMalloc((void **)&cu_count, pdfsum_size_in_byte);
    cudaMalloc((void **)&cu_minweights, pdfsum_size_in_byte);


    /*zero*/
    cudaMemset((void *)cu_pdfmarginal, 0, pdfsum_size_in_byte);
    cudaMemset((void *)cu_count, 0, pdfsum_size_in_byte);
    cudaMemset((void *)cu_weightsum, 0, pdfsum_size_in_byte);
    cudaMemset((void *)cu_minweights, 0, pdfsum_size_in_byte);


    allGPUPdfMarginalAndWeight<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors, cu_knn, cu_pdfmarginal, cu_weightsum, cu_count, cu_minweights);
    cudaFree(cu_count);
    
    /*Allocate and initialize*/
    CUDAmalloc((void **)&cu_radiance, radiance_size_in_byte);
    CUDAmalloc((void **)&cu_tempRadiance, radiance_size_in_byte);
    cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
    cudaMemset((void *)cu_tempRadiance, 0, radiance_size_in_byte);
    int i=0;
    updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
    for (; i < iterations ; i++){
        cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
        allGPUScatterRadianceWithWeight<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfmarginal, cu_neighbors, cu_knn, cu_tempRadiance, cu_radiance, cu_weightsum, cu_minweights);
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.blur_results[i], radiance_size_in_byte);
        cudaDeviceSynchronize();
        updateWithOptDirectRadiance<<<blocknum, THREADDIM>>>(cu_direct_radiance, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
        lastRun<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.mc_results[i], radiance_size_in_byte);
    }
    cudaFree(cu_radiance);
    cudaFree(cu_tempRadiance);
    cudaFree(cu_pdfmarginal);
    cudaFree(cu_weightsum);
    cudaFree(cu_minweights);
    
}


void computeScatterAllOnGPUWithDirectOptandWeight(const SPoint *cu_sps,const int *cu_neighbors, 
    const int *cu_knn, const int * cu_dknn, int sizeofsp, int iterations, int k, int dk, val3f * cu_direct_radiance, ResultSpace ret, const size_t pitch_n, bool jitter){
    std::cout<<"starting computing scattering radiance!"<<std::endl;
    size_t pdfsum_size_in_byte = sizeofsp * sizeof(float);
    size_t radiance_size_in_byte = sizeofsp * sizeof(val3f);
    float *cu_pdfmarginal, *cu_weightsum, *cu_minweights, *cu_count;
    val3f *cu_tempRadiance, *cu_radiance;    
    
    /*copy constant to symbol*/
    printf("the maxk k I use for indirect: %d", k);
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_pitch_n, &pitch_n, sizeof(size_t));

    /*compute marginal pdf sum*/
    int blocknum = sizeofsp / THREADDIM + 1;
    cudaMalloc((void **)&cu_pdfmarginal, pdfsum_size_in_byte);
    cudaMalloc((void **)&cu_weightsum, pdfsum_size_in_byte);
    cudaMalloc((void **)&cu_count, pdfsum_size_in_byte);
    cudaMalloc((void **)&cu_minweights, pdfsum_size_in_byte);

    /*zero*/
    cudaMemset((void *)cu_pdfmarginal, 0, pdfsum_size_in_byte);
    cudaMemset((void *)cu_count, 0, pdfsum_size_in_byte);
    cudaMemset((void *)cu_weightsum, 0, pdfsum_size_in_byte);
    cudaMemset((void *)cu_minweights, 0, pdfsum_size_in_byte);
    allGPUPdfMarginalAndWeight<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors, cu_knn, cu_pdfmarginal, cu_weightsum, cu_count, cu_minweights);
    cudaFree(cu_count);
    
    /*Allocate and initialize*/
    CUDAmalloc((void **)&cu_radiance, radiance_size_in_byte);
    CUDAmalloc((void **)&cu_tempRadiance, radiance_size_in_byte);
    cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
    cudaMemset((void *)cu_tempRadiance, 0, radiance_size_in_byte);
    int i=0;
    updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
    for (; i < iterations ; i++){
        cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
        allGPUScatterRadianceWithWeight<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfmarginal, cu_neighbors, cu_knn, cu_tempRadiance, cu_radiance, cu_weightsum, cu_minweights);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();

        if(i==iterations-1) CUDAcpyD2H((void *)cu_radiance, (void *)ret.blur_results[0], radiance_size_in_byte);
       
        updateWithOptDirectRadiance<<<blocknum, THREADDIM>>>(cu_direct_radiance, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        
        if(i==iterations-1){
            cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
            if(jitter){
                cudaMemset((void *)cu_pdfmarginal, 0, pdfsum_size_in_byte);
                if (dk <= k){
                    allGPUPdfMarginalJitter<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors, cu_knn, cu_pdfmarginal);
                    lastRunJitter<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfmarginal, cu_neighbors, cu_knn, cu_tempRadiance, cu_radiance);
                }else{
                    cudaMemcpyToSymbol(d_k, &dk, sizeof(int));
                    allGPUPdfMarginalJitter<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors, cu_dknn, cu_pdfmarginal);
                    lastRunJitter<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfmarginal, cu_neighbors, cu_dknn, cu_tempRadiance, cu_radiance);
                } 
            }else{
                lastRun<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance);
            }
            cudaDeviceSynchronize();
            clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
            cudaDeviceSynchronize();
            CUDAcpyD2H((void *)cu_radiance, (void *)ret.mc_results[0], radiance_size_in_byte);
        } 
    }
    cudaFree(cu_radiance);
    cudaFree(cu_tempRadiance);
    cudaFree(cu_pdfmarginal);
    cudaFree(cu_weightsum);
    cudaFree(cu_minweights);
}

/* wraped functions related to recompute bsdf from cpp*/
void hashKNN(SPoint *cu_sps, GridEntry * gridentries, int *cellinfo, hparam hash_param, int *cu_neighbors, int sizeofsp, int *cu_knn){
    
    float *cu_dists;
    float maxRadius = 3 * max(hash_param.cellsize[0], max(hash_param.cellsize[1], hash_param.cellsize[2]));
    KPara kp={
        {hash_param.minb[0], hash_param.minb[1], hash_param.minb[2]},
        {hash_param.cellsize[0], hash_param.cellsize[1], hash_param.cellsize[2]},
        {hash_param.dim[0], hash_param.dim[1], hash_param.dim[2]},
        hash_param.k,
        hash_param.dk,
        std::max(hash_param.k, hash_param.dk),
        maxRadius
    };

    int blocknum = sizeofsp / THREADDIM + 1;
    int gridnum = hash_param.dim[0]*hash_param.dim[1]*hash_param.dim[2];
    cudaMemcpyToSymbol(d_grid_size, &gridnum, sizeof(int));

    cudaMemcpyToSymbol(d_params, (void *)&kp, sizeof(KPara));
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    

    size_t size_float_in_byte = sizeofsp * hash_param.k * sizeof(float);
    CUDAmalloc((void **)&cu_dists, size_float_in_byte);
    
    NearestNeighbor<<<blocknum,THREADDIM>>>(cu_sps, gridentries, cellinfo, cu_dists, cu_neighbors, cu_knn);
    cudaDeviceSynchronize();
    cudaFree(cu_dists);
}




// void batchHashCluster(SPoint *cu_sps, GridEntry * gridentries, int *cellinfo, hparam hash_param, int *cu_clusters, int sizeofsp, int maxCPG, int sizeofcluster, int *cu_cluster_i, int *cu_clustermember_count){
//     float maxRadius = 3 * max(hash_param.cellsize[0], max(hash_param.cellsize[1], hash_param.cellsize[2]));
//     KPara kp={
//         {hash_param.minb[0], hash_param.minb[1], hash_param.minb[2]},
//         {hash_param.cellsize[0], hash_param.cellsize[1], hash_param.cellsize[2]},
//         {hash_param.dim[0], hash_param.dim[1], hash_param.dim[2]},
//         hash_param.k,
//         hash_param.dk,
//         std::max(hash_param.k, hash_param.dk) * 2,
//         maxRadius
//     };
//     cudaMemcpyToSymbol(d_params, (void *)&kp, sizeof(KPara));
//     cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
//     int gridnum = hash_param.dim[0]*hash_param.dim[1]*hash_param.dim[2];
//     cudaMemcpyToSymbol(d_grid_size, &gridnum, sizeof(int));

//     float *cu_dists;
//     int * cu_index;

//     int maxMem = maxCPG * 9;
//     int maxMemInKByte = maxMem * (sizeof(float) + sizeof(int)) / 1024;
//     /*Here I set the memory bound to be 4GB*/
//     int numThreadPKernel = 3 * 1024 * 1024 / maxMemInKByte;
//     int numBlock = numThreadPKernel / THREADDIM + 1;
//     int num_of_sp_computed = 0;

//     std::cout<<"\nmax points number:"<<maxCPG<<std::endl;
//     std::cout<<"\nmaxMemInKByte:"<<maxMemInKByte<<std::endl;
//     std::cout<<"\n numThreadPKernel:"<<numThreadPKernel<<std::endl;
//     std::cout<<"\n numBlock:"<<numBlock<<std::endl;
//     std::cout<<"\n num_of_sp_computed:"<<num_of_sp_computed<<std::endl;

//     cudaMemcpyToSymbol(d_num_thread_per_kernel, &numThreadPKernel, sizeof(int));
//     cudaMemcpyToSymbol(d_maxMem, &maxMem, sizeof(int));
//     cudaMemcpyToSymbol(d_blocknum, &numBlock, sizeof(int));

//     size_t pitch_dist;
//     size_t pitch_index;
//     int status = cudaMallocPitch(&cu_dists, &pitch_dist, maxMem * sizeof(float), numThreadPKernel);
//     CUDAErrorLog(status, "malloc pitch distances");
//     status = cudaMallocPitch(&cu_index, &pitch_index, maxMem * sizeof(int), numThreadPKernel);
//     CUDAErrorLog(status, "malloc pitch index");
//     cudaMemset((void *)cu_clustermember_count, 0, sizeofcluster * sizeof(int));
//     for(num_of_sp_computed = 0 ; num_of_sp_computed < sizeofsp ; num_of_sp_computed += numThreadPKernel){
//         cudaMemcpyToSymbol(d_index_offset, &num_of_sp_computed, sizeof(int));
//         batchCluster<<<numBlock,THREADDIM>>>(cu_sps, gridentries, cellinfo, cu_dists, cu_index, cu_clusters, pitch_dist, pitch_index, cu_clustermember_count, cu_cluster_i);
//         cudaDeviceSynchronize();
//     }
//     cudaFree(cu_dists);
//     cudaFree(cu_index);
// }


void batchHashKNN(SPoint *cu_sps, GridEntry * gridentries, int *cellinfo, hparam hash_param, int *cu_neighbors, 
    int sizeofsp, int maxCPG, int *cu_knn, int *cu_dknn, const size_t pitch_n){
    //
    float maxRadius = 3 * max(hash_param.cellsize[0], max(hash_param.cellsize[1], hash_param.cellsize[2]));
    KPara kp={
        {hash_param.minb[0], hash_param.minb[1], hash_param.minb[2]},
        {hash_param.cellsize[0], hash_param.cellsize[1], hash_param.cellsize[2]},
        {hash_param.dim[0], hash_param.dim[1], hash_param.dim[2]},
        hash_param.k,
        hash_param.dk,
        std::max(hash_param.k, hash_param.dk),
        maxRadius
    };
    cudaMemcpyToSymbol(d_params, (void *)&kp, sizeof(KPara));
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_pitch_n, &pitch_n, sizeof(size_t));
    int gridnum = hash_param.dim[0]*hash_param.dim[1]*hash_param.dim[2];
    cudaMemcpyToSymbol(d_grid_size, &gridnum, sizeof(int));

    float *cu_dists;
    int * cu_index;
    // size_t size_float_in_byte = sizeofsp * k * sizeof(float);
    int maxMem = maxCPG * 8;
    int maxMemInKByte = maxMem * (sizeof(float) + sizeof(int)) / 1024;
    /*Here I set the memory bound to be 4GB*/
    size_t free_t, total_t;
    cudaMemGetInfo(&free_t, &total_t);
    printf("\t\t avaliable space %.3f mb\n", free_t / (1048576.0));
    int numThreadPKernel = ( (size_t) floor (free_t * 0.5) / 1024 ) / maxMemInKByte;

    if(numThreadPKernel > sizeofsp)
        numThreadPKernel = sizeofsp + 1;

    int numBlock = numThreadPKernel / THREADDIM + 1;
    int num_of_sp_computed = 0;

    std::cout<<"\nmax points number:"<<maxCPG<<std::endl;
    std::cout<<"\nmaxMemInKByte:"<<maxMemInKByte<<std::endl;
    std::cout<<"\n numThreadPKernel:"<<numThreadPKernel<<std::endl;
    std::cout<<"\n numBlock:"<<numBlock<<std::endl;
    std::cout<<"\n num_of_sp_computed:"<<num_of_sp_computed<<std::endl;

    cudaMemcpyToSymbol(d_num_thread_per_kernel, &numThreadPKernel, sizeof(int));
    cudaMemcpyToSymbol(d_maxMem, &maxMem, sizeof(int));
    cudaMemcpyToSymbol(d_blocknum, &numBlock, sizeof(int));

    size_t pitch_dist;
    size_t pitch_index;
    int status = cudaMallocPitch(&cu_dists, &pitch_dist, maxMem * sizeof(float), numThreadPKernel);
    CUDAErrorLog(status, "malloc pitch distances");
    status = cudaMallocPitch(&cu_index, &pitch_index, maxMem * sizeof(int), numThreadPKernel);
    CUDAErrorLog(status, "malloc pitch index");


    for(num_of_sp_computed = 0 ; num_of_sp_computed < sizeofsp ; num_of_sp_computed += numThreadPKernel){
        cudaMemcpyToSymbol(d_index_offset, &num_of_sp_computed, sizeof(int));
        batchNearestNeighbor<<<numBlock,THREADDIM>>>(cu_sps, gridentries, cellinfo, cu_dists, cu_index, cu_neighbors, cu_knn, cu_dknn, pitch_dist, pitch_index);
        cudaDeviceSynchronize();
    }
    cudaFree(cu_dists);
    cudaFree(cu_index);
    std::cout<<"\n succeed!!!"<<std::endl;
}

void CUDARadiance(SPoint *sps, val3f *cu_result, int num){
    /*sps and cu_result are all on cuda already*/
    val3f *cu_bsdf;
    float *cu_pdf;
    size_t size_in_byte = num * sizeof(val3f);
    CUDAmalloc((void **)&cu_bsdf, size_in_byte);
    CUDAmalloc((void **)&cu_pdf, num * sizeof(float));

    cudaMemcpyToSymbol(d_sizeofsp, &num, sizeof(int));

    cudaDeviceSynchronize();
    int blocknum = num / THREADDIM + 1;
    bsdfeval<<<blocknum, THREADDIM>>>(sps, cu_bsdf);
    cudaDeviceSynchronize();
    std::cout<<"bsdf computed!"<<std::endl;
    pdfeval<<<blocknum, THREADDIM>>>(sps, cu_pdf);
    std::cout<<"pdf computed!"<<std::endl;
    computeRadiance<<<blocknum, THREADDIM>>>(sps, cu_bsdf, cu_pdf, cu_result);
    cudaDeviceSynchronize();
    std::cout<<"radiance computed!"<<std::endl;
    CUDAdelete(cu_bsdf);
    CUDAdelete(cu_pdf);
}

void computeMISAllOnGPURecord(const SPoint *cu_sps, const int * cu_neighbors, const int *cu_knn, int sizeofsp, int iter, int k, ResultSpace ret, const size_t pitch_n){
    float *cu_pdfsum;
    size_t pdfsum_size_in_byte = k * sizeofsp * sizeof(float);
    size_t size_radiance_in_byte = sizeofsp * sizeof(val3f);
    val3f *cu_tempRadiance, *cu_radiance;    
    cudaMalloc((void **)&cu_pdfsum, pdfsum_size_in_byte);
    cudaMemset((void *)cu_pdfsum, 0, pdfsum_size_in_byte);

    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_pitch_n, &pitch_n, sizeof(size_t));

    int blocknum = sizeofsp / THREADDIM + 1;
    allGPUPdfSum<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors, cu_knn, cu_pdfsum);

    cudaMalloc((void **)&cu_tempRadiance, size_radiance_in_byte);
    cudaMalloc((void **)&cu_radiance, size_radiance_in_byte);
    cudaMemset((void *)cu_radiance, 0, size_radiance_in_byte);
    cudaMemset((void *)cu_tempRadiance, 0, size_radiance_in_byte);

    int i=0;
    updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
    cudaDeviceSynchronize();
    for (; i < iter ; i++){
        cudaMemset((void *)cu_radiance, 0, size_radiance_in_byte);
        allGPUMISRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfsum, cu_neighbors, cu_knn, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.blur_results[i], size_radiance_in_byte);
        updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i+1);
        cudaDeviceSynchronize();
        cudaMemset((void *)cu_radiance, 0, size_radiance_in_byte);
        lastRun<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.mc_results[i], size_radiance_in_byte);
    }

    cudaFree(cu_tempRadiance);
    cudaFree(cu_pdfsum);
    cudaFree(cu_radiance);
}


void computeMISAllOnGPU(const SPoint *cu_sps, const int * cu_neighbors, const int *cu_knn, const int *cu_dknn, 
    int sizeofsp, int iter, int k, int dk, ResultSpace ret, const size_t pitch_n, bool jitter){
    float *cu_pdfsum;
    size_t pdfsum_size_in_byte = k * sizeofsp * sizeof(float);
    size_t size_radiance_in_byte = sizeofsp * sizeof(val3f);
    val3f *cu_tempRadiance, *cu_radiance;    
    cudaMalloc((void **)&cu_pdfsum, pdfsum_size_in_byte);
    cudaMemset((void *)cu_pdfsum, 0, pdfsum_size_in_byte);

    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_k, &k, sizeof(int));
    cudaMemcpyToSymbol(d_pitch_n, &pitch_n, sizeof(size_t));

    int blocknum = sizeofsp / THREADDIM + 1;
    allGPUPdfSum<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors, cu_knn, cu_pdfsum);

    cudaMalloc((void **)&cu_tempRadiance, size_radiance_in_byte);
    cudaMalloc((void **)&cu_radiance, size_radiance_in_byte);
    cudaMemset((void *)cu_radiance, 0, size_radiance_in_byte);
    cudaMemset((void *)cu_tempRadiance, 0, size_radiance_in_byte);

    int i=0;
    updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
    cudaDeviceSynchronize();
    for (; i < iter ; i++){
        cudaMemset((void *)cu_radiance, 0, size_radiance_in_byte);
        allGPUMISRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfsum, cu_neighbors, cu_knn, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();

        if (i == iter-1) CUDAcpyD2H((void *)cu_radiance, (void *)ret.blur_results[0], size_radiance_in_byte);
        updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
        cudaDeviceSynchronize();

        if (i == iter-1) {
            cudaMemset((void *)cu_radiance, 0, size_radiance_in_byte);
            if(jitter){
                cudaMemset((void *)cu_pdfsum, 0, pdfsum_size_in_byte);
                if(dk <= k){
                    allGPUPdfSumJitter<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors, cu_knn, cu_pdfsum);
                    allGPUMISRadianceJitter<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfsum, cu_neighbors, cu_knn, cu_tempRadiance, cu_radiance);
                }else{
                    allGPUPdfSumJitter<<<blocknum, THREADDIM>>>(cu_sps, cu_neighbors, cu_dknn, cu_pdfsum);
                    allGPUMISRadianceJitter<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfsum, cu_neighbors, cu_dknn, cu_tempRadiance, cu_radiance);
                }
            }else{
                lastRun<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance);
            }
            cudaDeviceSynchronize();
            clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
            cudaDeviceSynchronize();
            CUDAcpyD2H((void *)cu_radiance, (void *)ret.mc_results[0], size_radiance_in_byte);
        }
    }
    cudaFree(cu_tempRadiance);
    cudaFree(cu_pdfsum);
    cudaFree(cu_radiance);
}



void ClusterScatterRecord(SPoint *cu_sps, int numClusters, const int *cu_clusters, 
    int *cu_np_in_clusters, int sizeofsp, int iterations, ResultSpace &ret){
    // std::cout<<"starting computing scattering radiance!"<<std::endl;
    size_t marginalpdf_size_in_byte = sizeofsp * sizeof(float);
    size_t radiance_size_in_byte = sizeofsp * sizeof(val3f);
    float *cu_pdfmarginal;
    val3f *cu_tempRadiance, *cu_radiance;    
    cudaMalloc((void **)&cu_pdfmarginal, marginalpdf_size_in_byte);

    thrust::device_ptr<int> dptr_cu_offsets(cu_np_in_clusters);
    int numclusterpoint = dptr_cu_offsets[numClusters-1];
    thrust::exclusive_scan(thrust::device, dptr_cu_offsets, numClusters + dptr_cu_offsets, dptr_cu_offsets);
    numclusterpoint = dptr_cu_offsets[numClusters-1] + numclusterpoint;

    // std::cout<<" num clusters "<<numclusterpoint<<std::endl;
    
    /*copy constant to symbol*/
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_num_clusters, &numClusters, sizeof(int));
    cudaMemcpyToSymbol(d_num_clusters_point, &numclusterpoint, sizeof(int));

    /*compute marginal pdf sum*/
    int blocknum = sizeofsp / THREADDIM + 1;
    int blocknum2 = numClusters / THREADDIM + 1;
    addtoSP<<<blocknum2, THREADDIM>>>(cu_sps, cu_clusters, cu_np_in_clusters);


    cudaMemset((void *)cu_pdfmarginal, 0, marginalpdf_size_in_byte);
    allGPUClusterPdfMarginal<<<blocknum, THREADDIM>>>(cu_sps, cu_clusters, cu_np_in_clusters, cu_pdfmarginal);
    cudaDeviceSynchronize();
    
    /*Allocate and initialize*/
    CUDAmalloc((void **)&cu_radiance, radiance_size_in_byte);
    CUDAmalloc((void **)&cu_tempRadiance, radiance_size_in_byte);
    cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
    cudaMemset((void *)cu_tempRadiance, 0, radiance_size_in_byte);

    int i=0;
    val3f *cu_cluster_out, *cu_cluster_in, *cu_cluster_ratio; 
    CUDAmalloc((void **)&cu_cluster_out, sizeof(val3f) * numClusters);
    CUDAmalloc((void **)&cu_cluster_in, sizeof(val3f) * numClusters);
    CUDAmalloc((void **)&cu_cluster_ratio, sizeof(val3f) * numClusters);
    cudaMemset((void *)cu_cluster_out, 0, sizeof(val3f) * numClusters);
    cudaMemset((void *)cu_cluster_in, 0, sizeof(val3f) * numClusters);
    cudaMemset((void *)cu_cluster_ratio, 0, sizeof(val3f) * numClusters);

    updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
    for (; i < iterations ; i++){
        cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
        allGPUClusterScatterRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfmarginal, cu_clusters, cu_np_in_clusters, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampCluster<<<blocknum, THREADDIM>>>(cu_sps, cu_cluster_out, cu_cluster_in, cu_radiance, cu_tempRadiance);
        cudaDeviceSynchronize();
        computeRatio<<<blocknum2, THREADDIM>>>(cu_sps, cu_cluster_out, cu_cluster_in, cu_cluster_ratio);
        cudaDeviceSynchronize();
        updateComputeCluster<<<blocknum, THREADDIM>>>(cu_sps, cu_radiance, cu_cluster_ratio);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.blur_results[i], radiance_size_in_byte);
        cudaDeviceSynchronize();
        updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i+1);
        cudaDeviceSynchronize();
        cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
        lastRun<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.mc_results[i], radiance_size_in_byte);
    }
    cudaFree(cu_radiance);
    cudaFree(cu_tempRadiance);
    cudaFree(cu_pdfmarginal);
    cudaFree(cu_cluster_out);
    cudaFree(cu_cluster_in);
}


void computeClusterScatterAllOnGPURecord(const SPoint *cu_sps, int numClusters, const int *cu_clusters, const int *cu_cluster_offset, int sizeofsp, int iterations, ResultSpace &ret){
    std::cout<<"starting computing scattering radiance!"<<std::endl;
    size_t marginalpdf_size_in_byte = sizeofsp * sizeof(float);
    size_t radiance_size_in_byte = sizeofsp * sizeof(val3f);
    float *cu_pdfmarginal;
    val3f *cu_tempRadiance, *cu_radiance;    
    cudaMalloc((void **)&cu_pdfmarginal, marginalpdf_size_in_byte);
    
    /*copy constant to symbol*/
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_num_clusters, &numClusters, sizeof(int));

    /*compute marginal pdf sum*/
    int blocknum = sizeofsp / THREADDIM + 1;
    int blocknum2 = numClusters / THREADDIM + 1;
    cudaMemset((void *)cu_pdfmarginal, 0, marginalpdf_size_in_byte);
    allGPUClusterPdfMarginal<<<blocknum, THREADDIM>>>(cu_sps, cu_clusters, cu_cluster_offset, cu_pdfmarginal);
    cudaDeviceSynchronize();
    
    /*Allocate and initialize*/
    CUDAmalloc((void **)&cu_radiance, radiance_size_in_byte);
    CUDAmalloc((void **)&cu_tempRadiance, radiance_size_in_byte);
    cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
    cudaMemset((void *)cu_tempRadiance, 0, radiance_size_in_byte);

    int i=0;
    val3f *cu_cluster_out, *cu_cluster_in, *cu_cluster_ratio; 
    CUDAmalloc((void **)&cu_cluster_out, sizeof(val3f) * numClusters);
    CUDAmalloc((void **)&cu_cluster_in, sizeof(val3f) * numClusters);
    CUDAmalloc((void **)&cu_cluster_ratio, sizeof(val3f) * numClusters);
    cudaMemset((void *)cu_cluster_out, 0, sizeof(val3f) * numClusters);
    cudaMemset((void *)cu_cluster_in, 0, sizeof(val3f) * numClusters);
    cudaMemset((void *)cu_cluster_ratio, 0, sizeof(val3f) * numClusters);

    updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
    for (; i < iterations ; i++){
        cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
        allGPUClusterScatterRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_pdfmarginal, cu_clusters, cu_cluster_offset, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampCluster<<<blocknum, THREADDIM>>>(cu_sps, cu_cluster_out, cu_cluster_in, cu_radiance, cu_tempRadiance);
        cudaDeviceSynchronize();
        computeRatio<<<blocknum2, THREADDIM>>>(cu_sps, cu_cluster_out, cu_cluster_in, cu_cluster_ratio);
        cudaDeviceSynchronize();
        updateComputeCluster<<<blocknum, THREADDIM>>>(cu_sps, cu_radiance, cu_cluster_ratio);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.blur_results[i], radiance_size_in_byte);
        cudaDeviceSynchronize();
        updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i+1);
        cudaDeviceSynchronize();
        cudaMemset((void *)cu_radiance, 0, radiance_size_in_byte);
        lastRun<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.mc_results[i], radiance_size_in_byte);
    }
    cudaFree(cu_radiance);
    cudaFree(cu_tempRadiance);
    cudaFree(cu_pdfmarginal);
    cudaFree(cu_cluster_out);
    cudaFree(cu_cluster_in);
}

void buildHashGridAndFindKNN(SPoint *cu_sps, hparam hp, int *cu_neighbors, int sizeofsp, int *cu_knn, int *cu_dknn, const size_t pitch_n){
    int num_grid = hp.dim[0] * hp.dim[1] * hp.dim[2];
    int * cu_hash_offset, * cu_poffsets;
    std::vector<int> offset(num_grid, 0);
    GridEntry * cu_hashgrid;
    size_t grid_int_size_in_byte = num_grid * sizeof(int);

    KPara kp={
        {hp.minb[0], hp.minb[1], hp.minb[2]},
        {hp.cellsize[0], hp.cellsize[1], hp.cellsize[2]},
        {hp.dim[0], hp.dim[1], hp.dim[2]},
        hp.k,
        hp.dk,
        std::max(hp.k, hp.dk),
        0
    };
    cudaMemcpyToSymbol(d_params, (void *)&kp, sizeof(KPara));
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_grid_size, &num_grid, sizeof(int));


    CUDAmalloc((void **)&cu_hash_offset, grid_int_size_in_byte);
    cudaMemset((void *)cu_hash_offset, 0, grid_int_size_in_byte);

    int blocknum = sizeofsp / THREADDIM + 1;
    int *data;
    
    data = (int *)malloc(grid_int_size_in_byte);
    if (data == nullptr)
        printf("!data is null!");

    /// build uniform grid ///
    HashGridSize<<<blocknum, THREADDIM>>>(cu_sps, cu_hash_offset);
    cudaDeviceSynchronize();
    CUDAcpyD2H(cu_hash_offset, data, grid_int_size_in_byte);
    // printf("\ntest init! num of grid = %d\n", num_grid);
    // printf("\ntest init! %d\n", data[0]);
    cudaFree(cu_hash_offset);
    int maxCPG = *std::max_element(data, data+num_grid);
    printf("\t [max] point per grid is :%d \n", maxCPG);
    thrust::exclusive_scan(data, data + num_grid, data);
    printf("\t [total] start %d, number :%d/%d \n", data[0], data[num_grid-1], sizeofsp);

    CUDAmalloc((void **)&cu_hash_offset, grid_int_size_in_byte);
    CUDAcpyH2D(cu_hash_offset, data, sizeof(int) * num_grid);

    CUDAmalloc((void **)&cu_poffsets, sizeofsp *sizeof(int));
    cudaMemset((void *)cu_poffsets, 0, sizeofsp *sizeof(int));
    getOffset<<<blocknum, THREADDIM>>>(cu_sps, cu_poffsets, cu_hash_offset);
    cudaDeviceSynchronize();
    cudaMemset(cu_hash_offset, 0, grid_int_size_in_byte);
    CUDAmalloc((void **)&cu_hashgrid, sizeof(GridEntry) * sizeofsp);
    cudaMemcpyToSymbol(d_grid_size, &num_grid, sizeof(int));
    cudaDeviceSynchronize();
    buildHash<<<blocknum, THREADDIM>>>(cu_sps, cu_poffsets, cu_hash_offset, cu_hashgrid);
    cudaDeviceSynchronize();
    cudaFree(cu_poffsets);
    CUDAcpyH2D(cu_hash_offset, data, sizeof(int) * num_grid);
    free(data);
    batchHashKNN(cu_sps, cu_hashgrid, cu_hash_offset, hp, cu_neighbors, sizeofsp, maxCPG, cu_knn, cu_dknn, pitch_n);
    cudaFree(cu_hashgrid);
    cudaFree(cu_hash_offset);
}

int countMyClusters(SPoint *cu_sps, int sizeofsp, hparam hp, int numClusters, int *cu_clusters, int *cu_np_in_clusters, int *cu_num_small_clusters){
    // int *smallest;
    // CUDAmalloc((void **)&smallest, sizeof(int));
    // thrust::device_ptr<int> dptr_smallest(smallest);

    // int *largest;
    // CUDAmalloc((void **)&largest, sizeof(int));
    // thrust::device_ptr<int> dptr_largest(largest);

    cudaMemcpyToSymbol(d_num_clusters, &numClusters, sizeof(int));

    thrust::device_ptr<int> dptr_cu_neighbors_offset(cu_np_in_clusters);
    int lastClusterSize = dptr_cu_neighbors_offset[numClusters-1];
    // dptr_smallest = thrust::min_element(dptr_cu_neighbors_offset, dptr_cu_neighbors_offset+numClusters);
    // dptr_largest = thrust::max_element(dptr_cu_neighbors_offset, dptr_cu_neighbors_offset+numClusters);
    // std::cout<<"[buildBatchClusters] max point in cluster: "<<dptr_largest[0]<<" min "<<dptr_smallest[0]<<std::endl;
    thrust::exclusive_scan(thrust::device, dptr_cu_neighbors_offset, dptr_cu_neighbors_offset + numClusters, dptr_cu_neighbors_offset);
    int num_of_cluster_point = lastClusterSize + dptr_cu_neighbors_offset[numClusters-1];
    //subdivide
    int *cu_num_in_cell;
    CUDAmalloc((void **)&cu_num_in_cell, sizeof(int) * (numClusters+1));
    cudaMemset(cu_num_in_cell, 0, sizeof(int) * (numClusters+1));
    cudaMemcpyToSymbol(d_num_clusters_point, &num_of_cluster_point, sizeof(int));

    int blocknum = sizeofsp / THREADDIM + 1;
    SaveClusters<<<blocknum, THREADDIM>>>(cu_sps, cu_np_in_clusters, cu_num_in_cell, cu_clusters);
    cudaDeviceSynchronize();
    int individuals;
    CUDAcpyD2H(cu_num_in_cell + numClusters, &individuals, sizeof(int));
    // std::cout<<"\n number of individual points: "<<individuals<<std::endl;
    cudaMemcpyToSymbol(d_individual, &individuals, sizeof(int));

    // subdivide once
    blocknum = numClusters / THREADDIM + 1;
    cudaMemset(cu_num_small_clusters, 0, sizeof(int) * numClusters);
    countClusters<<<blocknum, THREADDIM>>>(cu_num_in_cell, cu_num_small_clusters, hp.k);
    cudaDeviceSynchronize();
    thrust::device_ptr<int> dptr_num_small_clusters(cu_num_small_clusters);
    int lastSmallClusters = dptr_num_small_clusters[numClusters-1];
    thrust::exclusive_scan(thrust::device, dptr_num_small_clusters, dptr_num_small_clusters + numClusters, dptr_num_small_clusters);
    int newNumClusters = dptr_num_small_clusters[numClusters-1] + lastSmallClusters;
    // std::cout<<"\n number of old clusters"<<numClusters<<" number of new clusters: "<<newNumClusters<<" last element: "<<lastSmallClusters<<std::endl;
    return newNumClusters;
}

void subClusters(SPoint *cu_sps, 
                int sizeofsp, 
                hparam hp, 
                int numOldClusters,
                int numNewClusters, 
                int *cu_clusters, 
                const int *cu_clusters_old_offset, 
                int *cu_num_point_new_cluster,
                int *cu_num_small_clusters
                ){
    // int *smallest;
    // CUDAmalloc((void **)&smallest, sizeof(int));
    // thrust::device_ptr<int> dptr_smallest(smallest);

    // int *largest;
    // CUDAmalloc((void **)&largest, sizeof(int));
    // thrust::device_ptr<int> dptr_largest(largest);

    int blocknum = sizeofsp / THREADDIM + 1;    
    cudaMemset((void *)cu_num_point_new_cluster, 0, sizeof(int) * numNewClusters);
    SubdivideClusters<<<blocknum, THREADDIM>>>(cu_sps, 
                      cu_clusters, 
                      cu_clusters_old_offset, 
                      cu_num_point_new_cluster, 
                      cu_num_small_clusters, 
                      numNewClusters); 

    cudaDeviceSynchronize();

    // thrust::device_ptr<int> dptr_cu_num_point_new_cluster(cu_num_point_new_cluster);
    // dptr_largest = thrust::max_element(dptr_cu_num_point_new_cluster, dptr_cu_num_point_new_cluster + numNewClusters);
    // dptr_smallest = thrust::min_element(dptr_cu_num_point_new_cluster, dptr_cu_num_point_new_cluster + numNewClusters);
    // std::cout<<"Largest cluster: "<<dptr_largest[0]<<"Smallest cluster"<<dptr_smallest[0]<<std::endl;
    // // [validate] ================================================================

    // thrust::exclusive_scan(thrust::device, dptr_cu_num_point_new_cluster, dptr_cu_num_point_new_cluster+numNewClusters, dptr_cu_num_point_new_cluster); 
}

void FinalizeCluster(SPoint *cu_sps, int sizeofsp, hparam hp, int numClusters, int *cu_clusters, int *cu_np_in_clusters){
    cudaMemcpyToSymbol(d_num_clusters, &numClusters, sizeof(int));

    thrust::device_ptr<int> dptr_cu_neighbors_offset(cu_np_in_clusters);
    int lastClusterSize = dptr_cu_neighbors_offset[numClusters-1];
    // std::cout<<"last cluster size: "<<lastClusterSize;

    thrust::exclusive_scan(thrust::device, dptr_cu_neighbors_offset, dptr_cu_neighbors_offset + numClusters, dptr_cu_neighbors_offset);
    int numclusterpoint = lastClusterSize + dptr_cu_neighbors_offset[numClusters-1];

    int *cu_num_in_cell;
    cudaMemcpyToSymbol(d_num_clusters_point, &numclusterpoint, sizeof(int));
    int blocknum = sizeofsp / THREADDIM + 1;
    cudaMalloc((void **)&cu_num_in_cell, sizeof(int) * numClusters);
    cudaMemset(cu_num_in_cell, 0, sizeof(int) * numClusters);
    SaveClusters<<<blocknum, THREADDIM>>>(cu_sps, cu_np_in_clusters, cu_num_in_cell, cu_clusters);
    cudaDeviceSynchronize();
    // int blocknum2 = numClusters / THREADDIM + 1;
    // std::cout<<"checking clusters ... "<<std::endl;
    // CheckClusters<<<blocknum2, THREADDIM>>>(cu_np_in_clusters, cu_num_in_cell);
    // get number of clusters that only has 1 points
    CUDAdelete(cu_num_in_cell);
}

size_t MatrixElementsNumber(SPoint *cu_sps, const int *cu_offset, const int num_of_clusters, size_t * cu_elements_offset, int sizeofsp){
    size_t numberofelements;
    size_t lastpointneighbor;

    int blocknum = sizeofsp / THREADDIM + 1;
    cudaMemset(cu_elements_offset, 0, sizeof(size_t) * sizeofsp);
    CountNoneZeroElements<<<blocknum, THREADDIM>>>(cu_sps, cu_offset, cu_elements_offset);
    cudaDeviceSynchronize();
    thrust::device_ptr<size_t> dptr_cu_numberofNeighbor(cu_elements_offset);
    lastpointneighbor = dptr_cu_numberofNeighbor[sizeofsp-1];
    thrust::exclusive_scan(thrust::device, dptr_cu_numberofNeighbor, dptr_cu_numberofNeighbor+sizeofsp, dptr_cu_numberofNeighbor);
    numberofelements = dptr_cu_numberofNeighbor[sizeofsp-1]+lastpointneighbor;
    return numberofelements;
}

void ClusterIterations(SPoint *cu_sps, const int *cu_clusters, const int *cu_cluster_offset, const int num_of_clusters, const size_t *cu_element_offset, 
    int sizeofsp, const val3f * cu_elements, int iterations, ResultSpace &ret){
    val3f *cu_tempRadiance, *cu_radiance;    
    
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_num_clusters, &num_of_clusters, sizeof(int));

    /*compute marginal pdf sum*/
    int blocknum = sizeofsp / THREADDIM + 1;
    int blocknum2 = num_of_clusters / THREADDIM + 1; 
    /*Allocate and initialize*/
    CUDAmalloc((void **)&cu_radiance, sizeof(val3f) * sizeofsp);
    CUDAmalloc((void **)&cu_tempRadiance, sizeof(val3f) * sizeofsp);
    cudaMemset((void *)cu_radiance, 0, sizeof(val3f) * sizeofsp);
    cudaMemset((void *)cu_tempRadiance, 0, sizeof(val3f) * sizeofsp);

    int i=0;
    val3f *cu_cluster_out, *cu_cluster_in, *cu_cluster_ratio; 
    CUDAmalloc((void **)&cu_cluster_out, sizeof(val3f) * num_of_clusters);
    CUDAmalloc((void **)&cu_cluster_in, sizeof(val3f) * num_of_clusters);
    CUDAmalloc((void **)&cu_cluster_ratio, sizeof(val3f) * num_of_clusters);
    cudaMemset((void *)cu_cluster_out, 0, sizeof(val3f) * num_of_clusters);
    cudaMemset((void *)cu_cluster_in, 0, sizeof(val3f) * num_of_clusters);
    cudaMemset((void *)cu_cluster_ratio, 0, sizeof(val3f) * num_of_clusters);

    updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
    cudaDeviceSynchronize();
    for (; i < iterations ; i++){
        cudaMemset((void *)cu_radiance, 0, sizeof(val3f) * sizeofsp);
        MX<<<blocknum, THREADDIM>>>(cu_sps, cu_clusters, cu_elements, cu_cluster_offset, cu_element_offset, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampCluster<<<blocknum, THREADDIM>>>(cu_sps, cu_cluster_out, cu_cluster_in, cu_radiance, cu_tempRadiance);
        cudaDeviceSynchronize();
        computeRatio<<<blocknum2, THREADDIM>>>(cu_sps, cu_cluster_out, cu_cluster_in, cu_cluster_ratio);
        cudaDeviceSynchronize();
        updateComputeCluster<<<blocknum, THREADDIM>>>(cu_sps, cu_radiance, cu_cluster_ratio);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.blur_results[i], sizeof(val3f) * sizeofsp);
        cudaDeviceSynchronize();
        updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i+1);
        cudaDeviceSynchronize();
        cudaMemset((void *)cu_radiance, 0, sizeof(val3f) * sizeofsp);
        lastRun<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.mc_results[i], sizeof(val3f) * sizeofsp);
    }

    cudaFree(cu_radiance);
    cudaFree(cu_tempRadiance);
    cudaFree(cu_cluster_ratio);
    cudaFree(cu_cluster_out);
    cudaFree(cu_cluster_in);
}

void ClusterIterations2(SPoint *cu_sps, const int *cu_clusters, const int *cu_cluster_offset, const int num_of_clusters, const size_t *cu_element_offset, 
    int sizeofsp, const val3f * cu_elements,const val3f * cu_direct_radiance, int iterations, ResultSpace &ret){
        val3f *cu_tempRadiance, *cu_radiance;    
    
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_num_clusters, &num_of_clusters, sizeof(int));

    /*compute marginal pdf sum*/
    int blocknum = sizeofsp / THREADDIM + 1;
    int blocknum2 = num_of_clusters / THREADDIM + 1; 
    /*Allocate and initialize*/
    CUDAmalloc((void **)&cu_radiance, sizeof(val3f) * sizeofsp);
    CUDAmalloc((void **)&cu_tempRadiance, sizeof(val3f) * sizeofsp);
    cudaMemset((void *)cu_radiance, 0, sizeof(val3f) * sizeofsp);
    cudaMemset((void *)cu_tempRadiance, 0, sizeof(val3f) * sizeofsp);

    int i=0;
    val3f *cu_cluster_out, *cu_cluster_in, *cu_cluster_ratio; 
    CUDAmalloc((void **)&cu_cluster_out, sizeof(val3f) * num_of_clusters);
    CUDAmalloc((void **)&cu_cluster_in, sizeof(val3f) * num_of_clusters);
    CUDAmalloc((void **)&cu_cluster_ratio, sizeof(val3f) * num_of_clusters);
    cudaMemset((void *)cu_cluster_out, 0, sizeof(val3f) * num_of_clusters);
    cudaMemset((void *)cu_cluster_in, 0, sizeof(val3f) * num_of_clusters);
    cudaMemset((void *)cu_cluster_ratio, 0, sizeof(val3f) * num_of_clusters);

    updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
    cudaDeviceSynchronize();
    for (; i < iterations ; i++){
        cudaMemset((void *)cu_radiance, 0, sizeof(val3f) * sizeofsp);
        MX<<<blocknum, THREADDIM>>>(cu_sps, cu_clusters, cu_elements, cu_cluster_offset, cu_element_offset, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampCluster<<<blocknum, THREADDIM>>>(cu_sps, cu_cluster_out, cu_cluster_in, cu_radiance, cu_tempRadiance);
        cudaDeviceSynchronize();
        computeRatio<<<blocknum2, THREADDIM>>>(cu_sps, cu_cluster_out, cu_cluster_in, cu_cluster_ratio);
        cudaDeviceSynchronize();
        updateComputeCluster<<<blocknum, THREADDIM>>>(cu_sps, cu_radiance, cu_cluster_ratio);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.blur_results[i], sizeof(val3f) * sizeofsp);
        // cudaDeviceSynchronize();
        updateWithOptDirectRadiance<<<blocknum, THREADDIM>>>(cu_direct_radiance, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        cudaMemset((void *)cu_radiance, 0, sizeof(val3f) * sizeofsp);
        lastRun<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        CUDAcpyD2H((void *)cu_radiance, (void *)ret.mc_results[i], sizeof(val3f) * sizeofsp);
    }

    cudaFree(cu_radiance);
    cudaFree(cu_tempRadiance);
    cudaFree(cu_cluster_ratio);
    cudaFree(cu_cluster_out);
    cudaFree(cu_cluster_in);
}

void ClusterIterations3(SPoint *cu_sps, const int *cu_clusters, const int *cu_cluster_offset, const int num_of_clusters, const size_t *cu_element_offset, 
    int sizeofsp, const val3f * cu_elements,const val3f * cu_direct_radiance, int iterations, ResultSpace &ret){
        val3f *cu_tempRadiance, *cu_radiance;    
    
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_num_clusters, &num_of_clusters, sizeof(int));

    /*compute marginal pdf sum*/
    int blocknum = sizeofsp / THREADDIM + 1;
    int blocknum2 = num_of_clusters / THREADDIM + 1; 
    /*Allocate and initialize*/
    CUDAmalloc((void **)&cu_radiance, sizeof(val3f) * sizeofsp);
    CUDAmalloc((void **)&cu_tempRadiance, sizeof(val3f) * sizeofsp);
    cudaMemset((void *)cu_radiance, 0, sizeof(val3f) * sizeofsp);
    cudaMemset((void *)cu_tempRadiance, 0, sizeof(val3f) * sizeofsp);

    int i=0;
    val3f *cu_cluster_out, *cu_cluster_in, *cu_cluster_ratio; 
    CUDAmalloc((void **)&cu_cluster_out, sizeof(val3f) * num_of_clusters);
    CUDAmalloc((void **)&cu_cluster_in, sizeof(val3f) * num_of_clusters);
    CUDAmalloc((void **)&cu_cluster_ratio, sizeof(val3f) * num_of_clusters);
    cudaMemset((void *)cu_cluster_out, 0, sizeof(val3f) * num_of_clusters);
    cudaMemset((void *)cu_cluster_in, 0, sizeof(val3f) * num_of_clusters);
    cudaMemset((void *)cu_cluster_ratio, 0, sizeof(val3f) * num_of_clusters);

    updateRadiance<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance, i);
    cudaDeviceSynchronize();
    for (; i < iterations ; i++){
        cudaMemset((void *)cu_radiance, 0, sizeof(val3f) * sizeofsp);
        MX<<<blocknum, THREADDIM>>>(cu_sps, cu_clusters, cu_elements, cu_cluster_offset, cu_element_offset, cu_tempRadiance, cu_radiance);
        cudaDeviceSynchronize();
        clampCluster<<<blocknum, THREADDIM>>>(cu_sps, cu_cluster_out, cu_cluster_in, cu_radiance, cu_tempRadiance);
        cudaDeviceSynchronize();
        computeRatio<<<blocknum2, THREADDIM>>>(cu_sps, cu_cluster_out, cu_cluster_in, cu_cluster_ratio);
        cudaDeviceSynchronize();
        updateComputeCluster<<<blocknum, THREADDIM>>>(cu_sps, cu_radiance, cu_cluster_ratio);
        cudaDeviceSynchronize();
        clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
        cudaDeviceSynchronize();
        if(i == iterations-1){
            CUDAcpyD2H((void *)cu_radiance, (void *)ret.blur_results[i], sizeof(val3f) * sizeofsp);
        }else{
            updateWithOptDirectRadiance<<<blocknum, THREADDIM>>>(cu_direct_radiance, cu_tempRadiance, cu_radiance);
        }
        cudaDeviceSynchronize();        
    }
    cudaMemset((void *)cu_radiance, 0, sizeof(val3f) * sizeofsp);
    lastRun<<<blocknum, THREADDIM>>>(cu_sps, cu_tempRadiance, cu_radiance);
    cudaDeviceSynchronize();
    clampRadiance<<<blocknum, THREADDIM>>>(cu_radiance);
    cudaDeviceSynchronize();
    CUDAcpyD2H((void *)cu_radiance, (void *)ret.mc_results[iterations-1], sizeof(val3f) * sizeofsp);

    cudaFree(cu_radiance);
    cudaFree(cu_tempRadiance);
    cudaFree(cu_cluster_ratio);
    cudaFree(cu_cluster_out);
    cudaFree(cu_cluster_in);
}

void precomputedMatrixElemtns(SPoint *cu_sps, const int *cu_clusters, const int *cu_cluster_offset, const size_t *cu_element_offset, const int num_of_clusters, const int sizeofsp, const int numzeroelemetns, val3f *cu_matrix_elements){
    
    std::cout<<"starting computing scattering radiance!"<<std::endl;
    std::cout<<"number of zero elements! "<<numzeroelemetns<<std::endl;
    float *cu_pdfmarginal;
    cudaMalloc((void **)&cu_pdfmarginal, sizeofsp * sizeof(float));
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));
    cudaMemcpyToSymbol(d_num_clusters, &num_of_clusters, sizeof(int));
    
    /*compute marginal pdf sum*/
    int blocknum = sizeofsp / THREADDIM + 1;
    cudaMemset((void *)cu_pdfmarginal, 0, sizeofsp * sizeof(float));
    cudaMemset((void *)cu_matrix_elements, 0, numzeroelemetns * sizeof(val3f));
    allGPUClusterPdfMarginal<<<blocknum, THREADDIM>>>(cu_sps, cu_clusters, cu_cluster_offset, cu_pdfmarginal);
    cudaDeviceSynchronize();
    computeNoneZeroElements<<<blocknum, THREADDIM>>>(cu_sps, cu_clusters, cu_cluster_offset, cu_pdfmarginal, cu_element_offset, cu_matrix_elements);

    cudaFree(cu_pdfmarginal);
}

void buildBatchClusters(SPoint *cu_sps, int sizeofsp, int tablecells, hparam hp, int numClusters, int * indices, int *cu_clusters, int *cu_np_in_clusters){
    int *cu_clusters_idx, *cu_hash_table_sizes;
    CUDAmalloc((void **)&cu_hash_table_sizes, sizeof(int) * tablecells);
    CUDAmalloc((void **)&cu_clusters_idx, sizeof(int) * numClusters);
    CUDAcpyH2D(cu_clusters_idx, indices, sizeof(int) * numClusters);

    KPara kp={
        {hp.minb[0], hp.minb[1], hp.minb[2]},
        {hp.cellsize[0], hp.cellsize[1], hp.cellsize[2]},
        {hp.dim[0], hp.dim[1], hp.dim[2]},
        hp.k,
        hp.dk,
        std::max(hp.k, hp.dk),
        0
    };

    float dist = sqrtf(hp.cellsize[0] * hp.cellsize[0] + hp.cellsize[1] * hp.cellsize[1] + hp.cellsize[2] * hp.cellsize[2]) * 2.0;
    cudaMemcpyToSymbol(d_min_dist, &dist, sizeof(float));
    cudaMemcpyToSymbol(d_params, (void *)&kp, sizeof(KPara));
    cudaMemcpyToSymbol(d_sizeofsp, &sizeofsp, sizeof(int));

    cudaMemcpyToSymbol(d_num_clusters, &numClusters, sizeof(int));
    cudaMemcpyToSymbol(d_table_size, &tablecells, sizeof(int));
    cudaMemset(cu_hash_table_sizes, 0, sizeof(int) * tablecells);

    int blocknum = numClusters / THREADDIM + 1;
    HashTableSize<<<blocknum, THREADDIM>>>(cu_sps, cu_clusters_idx, cu_hash_table_sizes);
    cudaDeviceSynchronize();
    thrust::device_ptr<int> dptr_hash_table_offsets(cu_hash_table_sizes);
    
    // int *smallest;
    // CUDAmalloc((void **)&smallest, sizeof(int));
    // thrust::device_ptr<int> dptr_smallest(smallest);

    // int *largest;
    // CUDAmalloc((void **)&largest, sizeof(int));
    // thrust::device_ptr<int> dptr_largest(largest);

    int lasttablesize = dptr_hash_table_offsets[tablecells-1];
    // std::cout<<"[buildBatchClusters] last hash table cell size: "<<lasttablesize<<std::endl;
    // dptr_smallest = thrust::min_element(dptr_hash_table_offsets, dptr_hash_table_offsets+tablecells);
    // dptr_largest = thrust::max_element(dptr_hash_table_offsets, dptr_hash_table_offsets+tablecells);
    // std::cout<<"[buildBatchClusters] max: "<<dptr_largest[0]<<" min "<<dptr_smallest[0]<<std::endl;

    thrust::exclusive_scan(thrust::device, dptr_hash_table_offsets, dptr_hash_table_offsets+tablecells, dptr_hash_table_offsets);
    int lastoffset =  dptr_hash_table_offsets[tablecells-1];
    // std::cout<<"[buildBatchClusters] largest offset: "<<lastoffset<<"hash table size: "<<lastoffset+lasttablesize<<"clusters "<<numClusters<<std::endl;
    
    GridEntry *cu_hash_table;
    int *cu_num_in_cell;
    CUDAmalloc((void **)&cu_num_in_cell, sizeof(int) * tablecells);
    CUDAmalloc((void **)&cu_hash_table, sizeof(GridEntry) * numClusters);
    // blocknum = tablecells / THREADDIM + 1;
    // initializeHashTable<<<blocknum, THREADDIM>>>(cu_hash_table);

    cudaMemset(cu_num_in_cell, 0, sizeof(int) * tablecells);
    blocknum = numClusters / THREADDIM + 1;
    buildHashSub<<<blocknum, THREADDIM>>>(cu_sps, cu_clusters_idx, cu_hash_table_sizes, cu_num_in_cell, cu_hash_table);
    cudaDeviceSynchronize();
    cudaFree(cu_num_in_cell);
    
    // find nearest point from K
    blocknum = sizeofsp / THREADDIM + 1;
    cudaMemset(cu_np_in_clusters, 0, sizeof(int) * numClusters);
    Cluster<<<blocknum, THREADDIM>>>(cu_sps, cu_hash_table_sizes, cu_hash_table, cu_np_in_clusters);
    cudaDeviceSynchronize();
    
    // [ validated ] =================================================================
    CUDAdelete(cu_clusters_idx);
    CUDAdelete(cu_hash_table);
    CUDAdelete(cu_hash_table_sizes);
}



