require 'nn'

local ViterbiWrapper, parent = torch.class('nn.ViterbiWrapper','nn.Module')

local function cloneMLP(mlp)
   local mlpc
   if torch.typename(mlp) == 'nn.Sequential' then
      mlpc = nn.Sequential()
      for i=1,mlp:size() do
         mlpc:add( cloneMLP(mlp:get(i)) )
      end
   elseif torch.typename(mlp) == 'nn.Parallel' then
      mlpc = nn.Parallel()
      for i=1,#mlp.modules do
         mlpc:add( cloneMLP(mlp.modules[i]) )
      end
   else
      local f = torch.MemoryFile():binary()
      f:writeObject(mlp)
      f:seek(1)
      mlpc = f:readObject()
      f:close()

   end
   if mlp.weight then
      mlpc.weight:set(mlp.weight)
   end
   if mlp.bias then
      mlpc.bias:set(mlp.bias)
   end
   return mlpc
end

function ViterbiWrapper:__init(mlp,nClass,maxSentenceLong)
   parent.__init(self)
   self.mlp = mlp
   self.nClass = nClass
   self.maxSentenceLong = maxSentenceLong or 0
   self.mlpfclones = {}
   for i=1,self.maxSentenceLong do
      self.mlpfclones[i] = cloneMLP(self.mlp)
      io.stdout:write('.')
      io.stdout:flush()
   end
   self.currentNerror = 0
end

function ViterbiWrapper:reset(stdv)
   self.mlp:reset(stdv)
end

function ViterbiWrapper:write(file)
   parent.write(self, file)
   file:writeObject(self.mlp)
   file:writeInt(self.nClass)
   file:writeInt(self.maxSentenceLong)
   file:writeInt(self.currentNerror)
end

function ViterbiWrapper:read(file)
   parent.read(self, file)
   self.mlp = file:readObject()
   self.nClass = file:readInt()
   self.maxSentenceLong = file:readInt()
   self.currentNerror = file:readInt()
   self.mlpfclones = {}
   self.currentNerror = 0
end

function ViterbiWrapper:doClones()
   for i=1,self.maxSentenceLong do
      self.mlpfclones[i] = cloneMLP(self.mlp)
      io.stdout:write('.')
      io.stdout:flush()
   end
end

function ViterbiWrapper:forward(input)
   local examples = input
   self.output:resize(self.nClass,#examples)
   self.output:fill(0)
   for i,example in ipairs(examples) do
      local out = self.mlp:forward(example)
      self.output:select(2,i):copy(out)
   end
   return self.output
end

function ViterbiWrapper:backward(input,gradOutput)
   local examples = input
   local ce = 0
   for i=1, gradOutput:size(2) do
      if gradOutput:select(2,i):max() > 0 then
	 local ex = input[i]
	 ce = ce + 1
	 if self.maxSentenceLong == 0 then
	    self.mlp:forward(ex)
	    self.mlp:backward(ex,gradOutput:select(2,i))
	 else
	    self.mlpfclones[ce]:forward(ex)
	    self.mlpfclones[ce]:zeroGradParameters()
	    self.mlpfclones[ce]:backward(ex,gradOutput:select(2,i))
	 end
      end
   end
   self.currentNerror = ce
end

function ViterbiWrapper:zeroGradParameters()
   if self.maxSentenceLong == 0 then
      self.mlp:zeroGradParameters()
   else
      for i=1,self.currentNerror do
	 self.mlpfclones[i]:zeroGradParameters()
      end
   end
end

function ViterbiWrapper:updateParameters(learningRate)
   for i=1,self.currentNerror do
      if self.maxSentenceLong == 0 then
   	 self.mlp:updateParameters(learningRate/self.currentNerror) 
      else
	 self.mlpfclones[i]:updateParameters(learningRate/self.currentNerror)
      end
   end
end
