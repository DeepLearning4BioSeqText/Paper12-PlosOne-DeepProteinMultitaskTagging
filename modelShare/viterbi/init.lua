require 'torch'
require 'nn'

require "libviterbi"

torch.include('viterbi','Viterbi.lua')
torch.include('viterbi','ViterbiWrapper.lua')
torch.include('viterbi','ViterbiCriterion.lua')


