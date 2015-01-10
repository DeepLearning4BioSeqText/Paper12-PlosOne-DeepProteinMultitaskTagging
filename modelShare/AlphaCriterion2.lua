require 'nn'


local AlphaCriterion, parent = torch.class('nn.AlphaCriterion2','nn.Criterion')

local function table2tensor(tbl)

----- for debug
--   for i=1, #tbl do
--     io.stdout:write(tbl[i] .. ",")
--   end
--   io.stdout:write("\n")

   local t = torch.Tensor(#tbl)
   local i = 0
   t:apply(function()
              i = i + 1
              return tbl[i]
           end)
   return t
end

function AlphaCriterion:__init(nc)
   parent.__init(self)
   self.viterbi = nn.Viterbi(nc)
   self.nClass = nc
   self.normalize = false
   self.gradInput = self.viterbi.gradInput
end


function AlphaCriterion:forward(input, target)
   local weight = target.weight or 1
   target = table2tensor(target)

fordgb = input -- for debug 
---print(input:size()[1])
---print(input:size()[2])

   local _,score = self.viterbi:computeAlpha(input)
   local scoreCorrect = self.viterbi:forwardCorrect(input, target)

   return weight * ( (-scoreCorrect) - (-score) )
end


function AlphaCriterion:backward(input, target)
   local weight = target.weight or 1
   target = table2tensor(target)
   
   self.viterbi:zeroGradInput(input)
   self.viterbi:zeroGradParameters()
   self.viterbi:backwardCorrect(input, target, -1)
   self.viterbi:backwardAlpha(input, 1)
   return self.gradInput:mul(weight)
end

function AlphaCriterion:write(file)
   parent.write(self, file)

   file:writeObject(self.viterbi)
   file:writeInt(self.nClass)
   file:writeBool(self.normalize)
   file:writeObject(self.gradInput)
end

function AlphaCriterion:read(file)
   parent.read(self, file)

   self.viterbi = file:readObject()
   self.nClass = file:readInt()
   self.normalize = file:readBool()
   self.gradInput = file:readObject()
end
