require 'nn'

----print('USING FASTER TRAINING SEQUENCE WRAPPER')

local SequenceWrapper, parent = torch.class('nn.SequenceWrapper','nn.Module')

function SequenceWrapper:__init(mlp,nClass,maxSentenceLong)
   parent.__init(self)
   self.mlp = mlp
   self.nClass = nClass
   self.maxSentenceLong = maxSentenceLong or 0
   self.mlpfclones = {}
   self.currentNError = 0
   for i=1,self.maxSentenceLong do
      self.mlpfclones[i] = self.mlp:clone('weight', 'bias')
      io.stdout:write('.')
      io.stdout:flush()
   end
end

function SequenceWrapper:reset(stdv)
   self.mlp:reset(stdv)
end

function SequenceWrapper:write(file)
   parent.write(self, file)
   file:writeObject(self.mlp)
   file:writeInt(self.nClass)
   file:writeInt(self.maxSentenceLong)
   file:writeObject(self.mlpfclones)
   file:writeInt(self.currentNError)
end

function SequenceWrapper:read(file)
   parent.read(self, file)
   self.mlp = file:readObject()
   self.nClass = file:readInt()
   self.maxSentenceLong = file:readInt()
   self.mlpfclones = file:readObject()
   self.currentNError = file:readInt()
end

-- input is supposed to be a table of table, where each sub table is in the correct format
-- of the input of mlp
-- the outout is matrix where each column is the network output scores for corresponding input
function SequenceWrapper:forward(input)
   self.output:resize(self.nClass,#input)

   -- run the set through mlp
   for i,input_i in ipairs(input) do
      self.mlpfclones[i] = self.mlpfclones[i] or self.mlp:clone('weight', 'bias')
      local output_i = self.mlpfclones[i]:forward(input_i)
      self.output:select(2,i):copy(output_i)
   end

   return self.output
end

-- input is supposed to be a table of tables, where each sub table is in the correct format
-- of the input of mlp
-- gradOutput is supposed to be a matrix with each column representing the gradient from viterbi
-- per each error
function SequenceWrapper:backward(input,gradOutput)

   for i,input_i in ipairs(input) do
      self.mlpfclones[i]:zeroGradParameters()
      if gradOutput:select(2,i):max() > 0 or gradOutput:select(2,i):min() < 0 then
         self.mlpfclones[i]:backward(input_i, gradOutput:select(2,i))
      end
   end

   self.currentNError = #input
end

function SequenceWrapper:zeroGradParameters()
end

function SequenceWrapper:updateParameters(learningRate)
   for i=1,self.currentNError do
      self.mlpfclones[i]:updateParameters(learningRate)
   end
end
