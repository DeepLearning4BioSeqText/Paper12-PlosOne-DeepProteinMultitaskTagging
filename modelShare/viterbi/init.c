#include "luaT.h"

extern void nn_Viterbi_init(lua_State *L);

DLL_EXPORT int luaopen_libviterbi(lua_State *L)
{
  lua_getfield(L, LUA_GLOBALSINDEX, "nn");

  nn_Viterbi_init(L);

  return 1;
}
