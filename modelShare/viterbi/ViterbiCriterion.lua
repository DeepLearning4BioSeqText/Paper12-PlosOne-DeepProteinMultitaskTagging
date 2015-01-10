require 'nn'

local ViterbiCriterion, parent = torch.class('nn.ViterbiCriterion','nn.Criterion')

function ViterbiCriterion:__init(nc)
   parent.__init(self)
   self.nClass = nc
   self.normalize = false
end


function ViterbiCriterion:forward(input,target)
   local nerror = 0
   for i=1,input:size(1) do
      if target[i] ~= input[i] then
	 nerror = nerror + 1
      end
   end
   self.output = nerror

   if self.normalize then
   end

   return self.output
end


function ViterbiCriterion:backward(input, target)
   self.gradInput:resize(self.nClass,input:size(1))
   self.gradInput:zero()
   for i=1,input:size(1) do
      if target[i] ~= input[i] then
	 self.gradInput[input[i]][i] = 1
	 self.gradInput[target[i]][i] = -1
      end
   end

   if self.normalize then
      self.gradInput:div(input:size(1))
   end

   return self.gradInput
end
