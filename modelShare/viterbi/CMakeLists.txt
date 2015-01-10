# List of C/C++ sources
SET(src init.c Viterbi.c)

# List of Lua sources
SET(luasrc init.lua Viterbi.lua ViterbiWrapper.lua ViterbiCriterion.lua) 

# Add a Torch package
ADD_TORCH_PACKAGE(viterbi "${src}" "${luasrc}")

TARGET_LINK_LIBRARIES(viterbi luaT TH)
