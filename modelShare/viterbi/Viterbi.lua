local Viterbi, parent = torch.class('nn.Viterbi','nn.Module')

function Viterbi:__init(size)
   parent.__init(self)

   self.transProb = torch.Tensor(size,size)
   self.startProb = torch.Tensor(size)
   self.gradTransProb = torch.Tensor(size,size)
   self.gradStartProb = torch.Tensor(size)
   self.alpha = torch.Tensor()
   self.beta = torch.Tensor()

   self:reset()
end

function Viterbi:reset()
   self.transProb:fill(-math.log(self.transProb:size(1)))
   self.startProb:fill(-math.log(self.startProb:size(1)))
end

function Viterbi:write(file)
   parent.write(self, file)

   file:writeObject(self.transProb)
   file:writeObject(self.startProb)
   file:writeObject(self.gradTransProb)
   file:writeObject(self.gradStartProb)
   file:writeObject(self.alpha)
   file:writeObject(self.beta)

end

function Viterbi:read(file)
   parent.read(self, file)

   self.transProb = file:readObject()
   self.startProb = file:readObject()
   self.gradTransProb = file:readObject()
   self.gradStartProb = file:readObject()
   self.alpha = file:readObject()
   self.beta = file:readObject()
end
