require "torch"
dofile "iolearn.lua"

-- global constants
maxAA1Index = 27
PSINum = 20
PSIchoice = '.NRF'    
maxLoad = -1

function createDataset( datadir, params ) 
   local home = datadir

   if params.psi and params.psi > 0 then 
         if params.psi == 1 then 
            PSIchoice = '.NR'  
         elseif params.psi == 2 then 
            PSIchoice = '.NRF'                 
         elseif params.psi == 3 then 
            PSIchoice = '.SP'
         else
            print("Wrong choice of parameter psi... change to default 2 (NRF) ")
      end 
   end

   --- now we build dataset
   local dataset = {}
   
   local aa1
   if params.a1sz and params.a1sz > 0 then 
      aa1 = iolearn.load(home .. "/aa1.dat", maxLoad) 
      dataset.iterator = iolearn.iterator(aa1)     
      print("[ Number of sequences:]  ", aa1:size())         
   end

   local psidata = {}   
   if params.psi and params.psi > 0 then  
      for curt = 1,PSINum do
            psidata[curt] = iolearn.load(home .."/psi"..tostring(curt)..".dat"..PSIchoice, maxLoad) 
      end
      dataset.iterator = iolearn.iterator(psidata[1])     
      print("[ Number of sequences:]  ", psidata[1]:size())         
   end

   -- transform sequences into windows
   if params.a1sz and params.a1sz > 0 then aa1 = iolearn.windows(aa1, params.niw, 1) end 
   if params.psi and params.psi > 0 then  
   for curt = 1,PSINum do
         psidata[curt] = iolearn.windows(psidata[curt], params.niw, -20) 
      end
      print("Number of windows extracted: " .. psidata[1]:size())      
   end
   
   if params.psi and params.psi > 0 and params.a1sz and params.a1sz > 0 then   
      if aa1:size() ~= psidata[1]:size() then  error('the dataset sucks') end
   end

   function dataset:size()
     return psidata[1]:size()
   end

   setmetatable(dataset, 
                {__index = function(self, index)                    
                    local input = {}
                    if params.a1sz and params.a1sz > 0 then  
                       table.insert(input, aa1[index])  
                       ----print(aa1[index])
                    end
                     if params.psi and params.psi > 0 then  
                       for curt = 1,PSINum do
                              curw = psidata[curt][index]
                              curw:apply(function(x)
                                              if x == -20 then
                                                  return -2
                                              elseif x <= -5 then
                                                  return -1
                                               elseif x >= 5 then
                                                  return 1
                                              else
                                                 return x/5
                                                  end  
                                          end)
                             table.insert(input, curw)
                       end
                    end
                    
                    return {input, 1}
                  end
                })
             
   return dataset 
end



function createSentenceDataset(dataset)
   local startSentenceIdx = {}
   local currentSentenceIdx
   local nSentence = 0

   print('[creating sentence dataset]')
   print('[original data size: ' .. dataset:size() .. ']')
   for i=1,dataset:size() do
      local sentenceIdx, wordIdx = unpack(dataset.iterator[i])
      if sentenceIdx ~= currentSentenceIdx then
         table.insert(startSentenceIdx, i)
         currentSentenceIdx = sentenceIdx
         nSentence = nSentence + 1
      end
   end
   print('[' .. nSentence .. ' sentences found]')

   local data = {}
   function data:size()
      return nSentence
   end
   setmetatable(data, {__index = function(self, index)
                                    local input = {}
                                    local label = {}
                                    local offset = startSentenceIdx[index]
                                    local sentenceIdx, wordIdx = unpack(dataset.iterator[offset])
                                    local i = 0
                                    
                                    while offset+i <= dataset:size() and dataset.iterator[offset+i][1] == sentenceIdx do
                                       local example = dataset[offset+i]
                                       table.insert(input, example[1])
                                       table.insert(label, example[2])
                                       i = i + 1
                                    end
                                    return {input, label}
                                 end})
   return data
end

function createWeightedDataset(dataset, norm)

   local data = {}
   function data:size()
      return dataset:size()
   end
   setmetatable(data, {__index = function(self, index)
                                    local example = dataset[index]
                                    local nword = #(example[2])

                                    if norm == 'word' then
                                       example[2].weight = 1
                                    elseif norm == 'sentence' then
                                       example[2].weight = 1/nword
                                    else
                                       error('unknown wanted normalization: ' .. norm)
                                    end
                                    return example
                                 end})

   return data
end

