#include "luaT.h"
#include "TH.h"

//#define expmx(x) exp(-(x))

//#define THTensor_get1d_FAST THTensor_get1d
#define THTensor_get1d_FAST(tensor, i0) tensor->storage->data[tensor->storageOffset+(i0)*tensor->stride[0]]
//#define THTensor_get2d_FAST THTensor_get2d
#define THTensor_get2d_FAST(tensor, i0, i1) tensor->storage->data[tensor->storageOffset+(i0)*tensor->stride[0]+(i1)*tensor->stride[1]]
//#define THTensor_get2d_FAST(tensor, i0, i1) tensor->storage->data[(i0)+(i1)*tensor->stride[1]]

inline float
expmx(float x)
{
#define EXACT_EXPONENTIAL 0
#if EXACT_EXPONENTIAL
  return exp(-x);
#else
  // fast approximation of exp(-x) for x positive
# define A0   (1.0)
# define A1   (0.125)
# define A2   (0.0078125)
# define A3   (0.00032552083)
# define A4   (1.0172526e-5)
  if (x < 13.0)
  {
//    assert(x>=0);
    float y;
    y = A0+x*(A1+x*(A2+x*(A3+x*A4)));
    y *= y;
    y *= y;
    y *= y;
    y = 1/y;
    return y;
  }
  return 0;
# undef A0
# undef A1
# undef A2
# undef A3
# undef A4
#endif
}

static const void* torch_Tensor_id = NULL;

static int nn_Viterbi_forward(lua_State *L)
{
  THTensor *emission   = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *transition = luaT_getfieldcheckudata(L, 1, "transProb", torch_Tensor_id);
  THTensor *init       = luaT_getfieldcheckudata(L, 1, "startProb", torch_Tensor_id);
  THTensor *path       = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);
  long N, T;
  float *delta, *deltap, *phi;
  long i, j, t;
 
  float probPath;

  luaL_argcheck(L, emission->nDimension == 2, 2, "invalid emission matrix");
  luaL_argcheck(L, transition->nDimension == 2, 2, "invalid transition matrix");
  luaL_argcheck(L, init->nDimension == 1, 2, "invalid init distribution matrix");
  
  N = emission->size[0];
  T = emission->size[1];

  luaL_argcheck(L, init->size[0] == N, 2, "incompatible number of states");
  luaL_argcheck(L, transition->size[0] == N, 2, "incompatible number of states");
  luaL_argcheck(L, transition->size[1] == N, 2, "transition matrix is not square");
  
  /* misc allocations */
  delta  = THAlloc(sizeof(float)*N);
  deltap = THAlloc(sizeof(float)*N);
  phi    = THAlloc(sizeof(float)*N*T);
  THTensor_resize1d(path, T);
  
  /* init */
  for(i = 0; i < N; i++)
    deltap[i] = THTensor_get1d_FAST(init, i) + THTensor_get2d_FAST(emission, i, 0);
  
  /* recursion */
  for(t = 1; t < T; t++)
  {
    float *deltan = delta;
    for(j = 0; j < N; j++)
    {
      float maxValue = -THInf;
      long maxIndex = 0;
      for(i = 0; i < N; i++)
      {
        float z = deltap[i] + THTensor_get2d_FAST(transition, i, j);
        if(z > maxValue)
        {
          maxValue = z;
          maxIndex = i;
        }
      }
      delta[j] = maxValue + THTensor_get2d_FAST(emission, j, t);
      phi[j+t*N] = maxIndex;
    }
    delta  = deltap;
    deltap = deltan;
  }

  {
    float maxValue = -THInf;
    long maxIndex = 0;
    for(j = 0; j < N; j++)
    {
      if(deltap[j] > maxValue)
      {
        maxValue = deltap[j];
        maxIndex = j;
      }
    }
    probPath = maxValue;
    THTensor_set1d(path, T-1, maxIndex+1); /* +1 for Lua */
  }

  for(t = T-2; t >= 0; t--)
    THTensor_set1d(path, t, phi[((long)(THTensor_get1d_FAST(path, t+1)))+(t+1)*N-1]+1); /* -1 and +1 for Lua */

  THFree(delta);
  THFree(deltap);
  THFree(phi);

  lua_pushnumber(L, probPath);
  return 2; /* path already on the stack and probPath */
}

static int nn_Viterbi_forwardCorrect(lua_State *L)
{
  THTensor *emission   = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *path       = luaT_checkudata(L, 3, torch_Tensor_id);
  THTensor *transition = luaT_getfieldcheckudata(L, 1, "transProb", torch_Tensor_id);
  THTensor *init       = luaT_getfieldcheckudata(L, 1, "startProb", torch_Tensor_id);
  long N, T;
  long i, j, t;
 
  float score;

  luaL_argcheck(L, emission->nDimension == 2, 2, "invalid emission matrix");
  luaL_argcheck(L, transition->nDimension == 2, 2, "invalid transition matrix");
  luaL_argcheck(L, init->nDimension == 1, 2, "invalid init distribution matrix");
  
  N = emission->size[0];
  T = emission->size[1];

  luaL_argcheck(L, init->size[0] == N, 2, "incompatible number of states");
  luaL_argcheck(L, transition->size[0] == N, 2, "incompatible number of states");
  luaL_argcheck(L, transition->size[1] == N, 2, "transition matrix is not square");
    
  /* init */
  j = THTensor_get1d_FAST(path, 0)-1;
  score = THTensor_get1d_FAST(init, j) + THTensor_get2d_FAST(emission, j, 0);
//  printf("C %f\n", THTensor_get2d_FAST(emission, j, 0));
//  printf("S %f\n", score);
  /* recursion */
  for(t = 1; t < T; t++)
  {
    i = THTensor_get1d_FAST(path, t-1)-1;
    j = THTensor_get1d_FAST(path, t)-1;
    score += THTensor_get2d_FAST(transition, i, j) + THTensor_get2d_FAST(emission, j, t);
//    printf("C %f\n", THTensor_get2d_FAST(emission, j, t));
//    printf("S %f\n", score);
  }

//  exit(0);
  lua_pushnumber(L, score);
  return 1; /* score */
}

static int nn_Viterbi_backwardCorrect(lua_State *L)
{
  THTensor *input   = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *path       = luaT_checkudata(L, 3, torch_Tensor_id);
  float g = luaL_optnumber(L, 4, -1); /* -1 because we MAXIMIZE */
  THTensor *gradInit       = luaT_getfieldcheckudata(L, 1, "gradStartProb", torch_Tensor_id);
  THTensor *gradTransition = luaT_getfieldcheckudata(L, 1, "gradTransProb", torch_Tensor_id);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);
  long N, T;
  int t;

  N = input->size[0];
  T = input->size[1];

/*  THTensor_resize2d(gradInput, N, T); */
/*  THTensor_zero(gradInput); should do that with zeroGradInput */ 

  for(t = 0; t < T; t++)
  {
    long it = (long)(THTensor_get1d_FAST(path, t)-1);
    THTensor_set2d(gradInput, it, t, THTensor_get2d_FAST(gradInput, it, t) + g);
    if(t > 0)
    {
      long itm1 = (long)(THTensor_get1d_FAST(path, t-1)-1);
      THTensor_set2d(gradTransition, itm1, it,
                     THTensor_get2d_FAST(gradTransition, itm1, it) + g);
    }
    else
    {
      THTensor_set1d(gradInit, it,
                     THTensor_get1d_FAST(gradInit, it) + g);
    }
  }
  return 1; /* gradInput already on the stack */
}

static int nn_Viterbi_alpha(lua_State *L)
{
  THTensor *emission   = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *transition = luaT_getfieldcheckudata(L, 1, "transProb", torch_Tensor_id);
  THTensor *init       = luaT_getfieldcheckudata(L, 1, "startProb", torch_Tensor_id);
  THTensor *alpha       = luaT_getfieldcheckudata(L, 1, "alpha", torch_Tensor_id);
  long N, T;
  long i, j, t;
 
  float prob;

  luaL_argcheck(L, emission->nDimension == 2, 2, "invalid emission matrix");
  luaL_argcheck(L, transition->nDimension == 2, 2, "invalid transition matrix");
  luaL_argcheck(L, init->nDimension == 1, 2, "invalid init distribution matrix");
  
  N = emission->size[0];
  T = emission->size[1];

  luaL_argcheck(L, init->size[0] == N, 2, "incompatible number of states");
  luaL_argcheck(L, transition->size[0] == N, 2, "incompatible number of states");
  luaL_argcheck(L, transition->size[1] == N, 2, "transition matrix is not square");
  
  /* misc allocations */
  THTensor_resize2d(alpha, N, T);
  
  /* init */
  for(i = 0; i < N; i++)
    THTensor_set2d(alpha, i, 0, THTensor_get1d_FAST(init, i) + THTensor_get2d_FAST(emission, i, 0));
  
  /* recursion */
  for(t = 1; t < T; t++)
  {
    for(j = 0; j < N; j++)
    {
      /* the max */
      float themax = -THInf;
      float sum = 0;
      for(i = 0; i < N; i++)
      {
        float z = THTensor_get2d_FAST(alpha, i, t-1) + THTensor_get2d_FAST(transition, i, j);
        if(z > themax)
          themax = z;
      }

      for(i = 0; i < N; i++)
        sum += expmx(themax-THTensor_get2d_FAST(alpha, i, t-1) - THTensor_get2d_FAST(transition, i, j));
      sum = themax + log(sum);

/* before: */
/*       float sum = THLogZero; */
/*       for(i = 0; i < N; i++) */
/*         sum = THLogAdd(sum, THTensor_get2d_FAST(alpha, i, t-1) + THTensor_get2d_FAST(transition, i, j)); */

      sum += THTensor_get2d_FAST(emission, j, t);
      THTensor_set2d(alpha, j, t, sum);
    }
  }
  
  {
    prob = THLogZero;
    for(j = 0; j < N; j++)
      prob = THLogAdd(prob, THTensor_get2d_FAST(alpha, j, T-1));
    lua_pushnumber(L, prob);
    lua_setfield(L, 1, "logProbability");
  }

  lua_pushnumber(L, prob);
  return 2; /* alpha already on the stack and prob */
}

static int nn_Viterbi_beta(lua_State *L)
{
  THTensor *emission   = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *transition = luaT_getfieldcheckudata(L, 1, "transProb", torch_Tensor_id);
  THTensor *init       = luaT_getfieldcheckudata(L, 1, "startProb", torch_Tensor_id);
  THTensor *beta       = luaT_getfieldcheckudata(L, 1, "beta", torch_Tensor_id);
  long N, T;
  long i, j, t;
 
  float prob;

  luaL_argcheck(L, emission->nDimension == 2, 2, "invalid emission matrix");
  luaL_argcheck(L, transition->nDimension == 2, 2, "invalid transition matrix");
  luaL_argcheck(L, init->nDimension == 1, 2, "invalid init distribution matrix");
  
  N = emission->size[0];
  T = emission->size[1];

  luaL_argcheck(L, init->size[0] == N, 2, "incompatible number of states");
  luaL_argcheck(L, transition->size[0] == N, 2, "incompatible number of states");
  luaL_argcheck(L, transition->size[1] == N, 2, "transition matrix is not square");
  
  /* misc allocations */
  THTensor_resize2d(beta, N, T);
  
  /* init */
  for(i = 0; i < N; i++)
    THTensor_set2d(beta, i, T-1, THLogOne);
  
  /* recursion */
  for(t = T-2; t >= 0; t--)
  {
    for(i = 0; i < N; i++)
    {
      float sum = THLogZero;
      for(j = 0; j < N; j++)
        sum = THLogAdd(sum, THTensor_get2d_FAST(beta, j, t+1) + THTensor_get2d_FAST(transition, i, j) + THTensor_get2d_FAST(emission, j, t+1));
      THTensor_set2d(beta, i, t, sum);
    }
  }
  
  return 1; /* beta already on the stack */
}

static int nn_Viterbi_zeroGradTransition(lua_State *L)
{
  THTensor *gradInit       = luaT_getfieldcheckudata(L, 1, "gradStartProb", torch_Tensor_id);
  THTensor *gradTransition = luaT_getfieldcheckudata(L, 1, "gradTransProb", torch_Tensor_id);
  THTensor *transition = luaT_getfieldcheckudata(L, 1, "transProb", torch_Tensor_id);
  long N;

  luaL_argcheck(L, transition->nDimension == 2, 2, "invalid transition matrix");
  N = transition->size[0];
  luaL_argcheck(L, transition->size[1] == N, 2, "transition matrix is not square");

  /* misc allocations */
  THTensor_resize1d(gradInit, N);
  THTensor_resize2d(gradTransition, N, N);
  THTensor_fill(gradInit, THLogZero);
  THTensor_fill(gradTransition, THLogZero);

  return 0;
}

static int nn_Viterbi_accTransition(lua_State *L)
{
  THTensor *emission   = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *transition = luaT_getfieldcheckudata(L, 1, "transProb", torch_Tensor_id);
  THTensor *init       = luaT_getfieldcheckudata(L, 1, "startProb", torch_Tensor_id);
  float logProbability = luaT_getfieldchecknumber(L, 1, "logProbability");
  THTensor *alpha      = luaT_getfieldcheckudata(L, 1, "alpha", torch_Tensor_id);
  THTensor *beta       = luaT_getfieldcheckudata(L, 1, "beta", torch_Tensor_id);
  THTensor *gradInit       = luaT_getfieldcheckudata(L, 1, "gradStartProb", torch_Tensor_id);
  THTensor *gradTransition = luaT_getfieldcheckudata(L, 1, "gradTransProb", torch_Tensor_id);
  long N, T;
  long i, j, t;
 
  float prob;

  luaL_argcheck(L, emission->nDimension == 2, 2, "invalid emission matrix");
  luaL_argcheck(L, transition->nDimension == 2, 2, "invalid transition matrix");
  luaL_argcheck(L, init->nDimension == 1, 2, "invalid init distribution matrix");
  luaL_argcheck(L, alpha->nDimension == 2, 2, "invalid alpha matrix");
  luaL_argcheck(L, beta->nDimension == 2, 2, "invalid beta matrix");
  
  N = emission->size[0];
  T = emission->size[1];

  luaL_argcheck(L, init->size[0] == N, 2, "incompatible number of states");
  luaL_argcheck(L, transition->size[0] == N, 2, "incompatible number of states");
  luaL_argcheck(L, transition->size[1] == N, 2, "transition matrix is not square");
  luaL_argcheck(L, alpha->size[0] == N && alpha->size[1] == T, 2, "invalid alpha matrix");
  luaL_argcheck(L, beta->size[0] == N && beta->size[1] == T, 2, "invalid beta matrix");
      
  /* transition */
  for(t = 0; t < T-1; t++)
  {
    for(j = 0; j < N; j++)
    {
      float z = THTensor_get2d_FAST(emission, j, t+1) + THTensor_get2d_FAST(beta, j, t+1) - logProbability;
      for(i = 0; i < N; i++)
      {
        THTensor_set2d(gradTransition, i, j, 
                       THLogAdd(
                         THTensor_get2d_FAST(gradTransition, i, j),
                         THTensor_get2d_FAST(alpha, i, t) + THTensor_get2d_FAST(transition, i, j) + z));
      }
    }
  }
  
  /* init */
  for(i = 0; i < N; i++)
  {
    THTensor_set1d(gradInit, i, 
                   THLogAdd(
                     THTensor_get1d_FAST(gradInit, i),
                     THTensor_get2d_FAST(alpha, i, 0) + THTensor_get2d_FAST(beta, i, 0) - logProbability));
  }
  
  return 2; /* gradInit and gradTransition already on the stack */
}

static int nn_Viterbi_updateTransition(lua_State *L)
{
  THTensor *transition = luaT_getfieldcheckudata(L, 1, "transProb", torch_Tensor_id);
  THTensor *init       = luaT_getfieldcheckudata(L, 1, "startProb", torch_Tensor_id);
  THTensor *gradInit       = luaT_getfieldcheckudata(L, 1, "gradStartProb", torch_Tensor_id);
  THTensor *gradTransition = luaT_getfieldcheckudata(L, 1, "gradTransProb", torch_Tensor_id);
  long N;
  long i, j;
 
  luaL_argcheck(L, transition->nDimension == 2, 2, "invalid transition matrix");
  luaL_argcheck(L, init->nDimension == 1, 2, "invalid init distribution matrix");
  luaL_argcheck(L, gradTransition->nDimension == 2, 2, "invalid gradTransition matrix");
  luaL_argcheck(L, gradInit->nDimension == 1, 2, "invalid gradInit distribution matrix");
  
  N = transition->size[0];

  luaL_argcheck(L, transition->size[0] == N && transition->size[1] == N, 2, "transition matrix is not square");
  luaL_argcheck(L, init->size[0] == N, 2, "invalid init matrix");
  luaL_argcheck(L, gradTransition->size[0] == N && gradTransition->size[1] == N, 2, "gradTransition matrix is not square");
  luaL_argcheck(L, gradInit->size[0] == N, 2, "invalid gradInit matrix");

  for(i = 0; i < N; i++)
  {
    float logSum = THLogZero;
    for(j = 0; j < N; j++)
      logSum = THLogAdd(logSum, THTensor_get2d_FAST(gradTransition, i, j));
    for(j = 0; j < N; j++)
      THTensor_set2d(transition, i, j, THTensor_get2d_FAST(gradTransition, i, j) - logSum);
  }

  float logSum = THLogZero;  
  for(i = 0; i < N; i++)
    logSum = THLogAdd(logSum, THTensor_get1d_FAST(gradInit, i));
  for(i = 0; i < N; i++)
    THTensor_set1d(init, i, THTensor_get1d_FAST(gradInit, i) - logSum);

  return 0;
}

static int nn_Viterbi_zeroGradInput(lua_State *L)
{
  THTensor *emission   = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);
  int N = emission->size[0];
  int T = emission->size[1];

  THTensor_resize2d(gradInput, N, T);
  THTensor_zero(gradInput);

  return 0;
}

static int nn_Viterbi_zeroGradParameters(lua_State *L)
{
  THTensor *gradInit       = luaT_getfieldcheckudata(L, 1, "gradStartProb", torch_Tensor_id);
  THTensor *gradTransition = luaT_getfieldcheckudata(L, 1, "gradTransProb", torch_Tensor_id);
  THTensor *transition = luaT_getfieldcheckudata(L, 1, "transProb", torch_Tensor_id);
  long N;

  luaL_argcheck(L, transition->nDimension == 2, 2, "invalid transition matrix");
  N = transition->size[0];
  luaL_argcheck(L, transition->size[1] == N, 2, "transition matrix is not square");

  /* misc allocations */
  THTensor_resize1d(gradInit, N);
  THTensor_resize2d(gradTransition, N, N);
  THTensor_zero(gradInit);
  THTensor_zero(gradTransition);

  return 0;
}

static int nn_Viterbi_backward(lua_State *L)
{
  THTensor *input   = luaT_checkudata(L, 2, torch_Tensor_id);
  float g = luaL_optnumber(L, 3, -1); /* -1 because we MAXIMIZE */
  THTensor *path       = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);
  THTensor *gradInit       = luaT_getfieldcheckudata(L, 1, "gradStartProb", torch_Tensor_id);
  THTensor *gradTransition = luaT_getfieldcheckudata(L, 1, "gradTransProb", torch_Tensor_id);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);
  long N, T;
  int t;

  N = input->size[0];
  T = input->size[1];

/*  THTensor_resize2d(gradInput, N, T); */
/*  THTensor_zero(gradInput); */

  for(t = 0; t < T; t++)
  {
    long it = (long)(THTensor_get1d_FAST(path, t)-1);
    THTensor_set2d(gradInput, it, t, THTensor_get2d_FAST(gradInput, it, t) + g);
    if(t > 0)
    {
      long itm1 = (long)(THTensor_get1d_FAST(path, t-1)-1);
      THTensor_set2d(gradTransition, itm1, it,
                     THTensor_get2d_FAST(gradTransition, itm1, it) + g);
    }
    else
    {
      THTensor_set1d(gradInit, it,
                     THTensor_get1d_FAST(gradInit, it) + g);
    }
  }
  return 1; /* gradInput already on the stack */
}

static void THTensor_dLogAdd(THTensor *tensor)
{
  float m = -THInf;
  float sum = 0;

  TH_TENSOR_APPLY(float, tensor, 
                  if(*tensor_p > m)
                    m = *tensor_p;
    );

  
  TH_TENSOR_APPLY(float, tensor, 
                  float z = expmx(m-*tensor_p); /* careful - here compared to exp */
                  *tensor_p = z;
                  sum += z;
    );

  THTensor_div(tensor, sum);
}

static int nn_Viterbi_backwardAlpha(lua_State *L)
{
  THTensor *input   = luaT_checkudata(L, 2, torch_Tensor_id);
  float g = luaL_optnumber(L, 3, -1); /* -1 because we MAXIMIZE */
  THTensor *alpha       = luaT_getfieldcheckudata(L, 1, "alpha", torch_Tensor_id);
  THTensor *gradInit       = luaT_getfieldcheckudata(L, 1, "gradStartProb", torch_Tensor_id);
  THTensor *transition = luaT_getfieldcheckudata(L, 1, "transProb", torch_Tensor_id);
  THTensor *gradTransition = luaT_getfieldcheckudata(L, 1, "gradTransProb", torch_Tensor_id);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);

  THTensor *tmp;
  THTensor *alpha_tm1;
  THTensor *gradAlpha_tm1;
  THTensor *gradAlpha_t;
  THTensor *gradInput_t;
  THTensor *transition_j;
  THTensor *gradTransition_j;

  long N, T;
  int j, t;

  N = input->size[0];
  T = input->size[1];

  tmp = THTensor_newWithSize1d(N);
  alpha_tm1 = THTensor_new();
  gradAlpha_tm1 = THTensor_newWithSize1d(N);
  gradAlpha_t = THTensor_newWithSize1d(N);
  gradInput_t = THTensor_new();
  transition_j = THTensor_new();
  gradTransition_j = THTensor_new();

/*  THTensor_resize2d(gradInput, N, T); */
/*  THTensor_zero(gradInput); */

  THTensor_select(alpha_tm1, alpha, 1, T-1);
  THTensor_copy(gradAlpha_t, alpha_tm1);
  THTensor_dLogAdd(gradAlpha_t);

  for(t = T-1; t > 0; t--)
  {
    THTensor_select(gradInput_t, gradInput, 1, t);
    THTensor_addTensor(gradInput_t, g, gradAlpha_t); /* -1 because we want to MAXIMIZE */

    THTensor_select(alpha_tm1, alpha, 1, t-1);
    THTensor_zero(gradAlpha_tm1);
    for(j = 0; j < N; j++)
    {
      THTensor_copy(tmp, alpha_tm1);
      THTensor_select(transition_j, transition, 1, j);
      THTensor_addTensor(tmp, 1, transition_j);
      THTensor_dLogAdd(tmp);
      THTensor_addTensor(gradAlpha_tm1, THTensor_get1d_FAST(gradAlpha_t, j), tmp);

      THTensor_select(gradTransition_j, gradTransition, 1, j);
      THTensor_addTensor(gradTransition_j, g*THTensor_get1d_FAST(gradAlpha_t, j), tmp); /* -1 because we want to MAXIMIZE */
    }
    
    {
      THTensor *z = gradAlpha_t;
      gradAlpha_t = gradAlpha_tm1;
      gradAlpha_tm1 = z;
    }
  }
  
  THTensor_select(gradInput_t, gradInput, 1, 0);
  THTensor_addTensor(gradInput_t, g, gradAlpha_t); /* -1 because we want to MAXIMIZE */
  THTensor_addTensor(gradInit, g, gradAlpha_t);    /* -1 because we want to MAXIMIZE */

  THTensor_free(tmp);
  THTensor_free(alpha_tm1);
  THTensor_free(gradAlpha_tm1);
  THTensor_free(gradAlpha_t);
  THTensor_free(gradInput_t);
  THTensor_free(transition_j);
  THTensor_free(gradTransition_j);

  return 1; /* gradInput already on the stack */
}

static int nn_Viterbi_updateParameters(lua_State *L)
{
  float learningRate = luaL_checknumber(L, 2);
  int doNormalize = luaT_optboolean(L, 3, 0);
  THTensor *transition = luaT_getfieldcheckudata(L, 1, "transProb", torch_Tensor_id);
  THTensor *init       = luaT_getfieldcheckudata(L, 1, "startProb", torch_Tensor_id);
  THTensor *gradInit       = luaT_getfieldcheckudata(L, 1, "gradStartProb", torch_Tensor_id);
  THTensor *gradTransition = luaT_getfieldcheckudata(L, 1, "gradTransProb", torch_Tensor_id);
  long N;
  long i, j;
 
  luaL_argcheck(L, transition->nDimension == 2, 2, "invalid transition matrix");
  luaL_argcheck(L, init->nDimension == 1, 2, "invalid init distribution matrix");
  luaL_argcheck(L, gradTransition->nDimension == 2, 2, "invalid gradTransition matrix");
  luaL_argcheck(L, gradInit->nDimension == 1, 2, "invalid gradInit distribution matrix");
  
  N = transition->size[0];

  luaL_argcheck(L, transition->size[0] == N && transition->size[1] == N, 2, "transition matrix is not square");
  luaL_argcheck(L, init->size[0] == N, 2, "invalid init matrix");
  luaL_argcheck(L, gradTransition->size[0] == N && gradTransition->size[1] == N, 2, "gradTransition matrix is not square");
  luaL_argcheck(L, gradInit->size[0] == N, 2, "invalid gradInit matrix");

  if(doNormalize)
  {
    /* Here we project on the constraints surface */

    THTensor *transition_i = THTensor_new();
    THTensor *gradTransition_i = THTensor_new();

    for(i = 0; i < N; i++)
    {
      float sumn = 0;
      float sumd = 0;
      THTensor_select(gradTransition_i, gradTransition, 0, i);
      THTensor_select(transition_i, transition, 0, i);
      
      TH_TENSOR_APPLY2(float, gradTransition_i, float, transition_i,
                       float z = exp(*transition_i_p);
                       sumn += *gradTransition_i_p * z;
                       sumd += z*z;);

      TH_TENSOR_APPLY2(float, gradTransition_i, float, transition_i,
                       *gradTransition_i_p -= sumn/sumd*exp(*transition_i_p););
    }

    {
      float sumn = 0;
      float sumd = 0;
      
      TH_TENSOR_APPLY2(float, gradInit, float, init,
                       float z = exp(*init_p);
                       sumn += *gradInit_p * z;
                       sumd += z*z;);

      TH_TENSOR_APPLY2(float, gradInit, float, init,
                       *gradInit_p -= sumn/sumd*exp(*init_p););
    }

    THTensor_free(transition_i);
    THTensor_free(gradTransition_i);
  }

  THTensor_addTensor(transition, -learningRate, gradTransition);
  THTensor_addTensor(init, -learningRate, gradInit);

  if(doNormalize)
  {
    /* si on veut normaliser... */
    for(i = 0; i < N; i++)
    {
      float logSum = THLogZero;
      for(j = 0; j < N; j++)
        logSum = THLogAdd(logSum, THTensor_get2d_FAST(transition, i, j));
      for(j = 0; j < N; j++)
        THTensor_set2d(transition, i, j, THTensor_get2d_FAST(transition, i, j) - logSum);
    }
    
    float logSum = THLogZero;  
    for(i = 0; i < N; i++)
      logSum = THLogAdd(logSum, THTensor_get1d_FAST(init, i));
    for(i = 0; i < N; i++)
      THTensor_set1d(init, i, THTensor_get1d_FAST(init, i) - logSum);
  }

  return 0;
}

static const struct luaL_Reg nn_Viterbi__ [] = {
  {"forward", nn_Viterbi_forward},
  {"computeAlpha", nn_Viterbi_alpha},
  {"backwardAlpha", nn_Viterbi_backwardAlpha},
  {"computeBeta", nn_Viterbi_beta},
  {"zeroGradTransition", nn_Viterbi_zeroGradTransition},
  {"accTransition", nn_Viterbi_accTransition},
  {"updateTransition", nn_Viterbi_updateTransition},
  {"backward", nn_Viterbi_backward},
  {"zeroGradParameters", nn_Viterbi_zeroGradParameters},
  {"updateParameters", nn_Viterbi_updateParameters},
  {"forwardCorrect", nn_Viterbi_forwardCorrect},
  {"backwardCorrect", nn_Viterbi_backwardCorrect},
  {"zeroGradInput", nn_Viterbi_zeroGradInput},
  {NULL, NULL}
};

void nn_Viterbi_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  luaT_newmetatable(L, "nn.Viterbi", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, nn_Viterbi__);
  lua_pop(L, 1);
}
