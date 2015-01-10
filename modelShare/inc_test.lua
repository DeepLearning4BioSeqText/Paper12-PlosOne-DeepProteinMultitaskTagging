
function test(mlp, dataset)
   local maxnum = dataset:size()       
   print('[testing over ' .. maxnum .. ' examples]')

   local predictions = torch.Tensor(maxnum)
   ---local scores = torch.Tensor(maxnum, params.numClasses)
   for t=1, maxnum do
      local example = dataset[t]
      local score = mlp:forward(example[1])
      ----print(score)
      local _, predClass = lab.max(score)      
      ----scores[t]:copy(score)
      predictions[t] = predClass[1]
   end
   print('[done]')
   return predictions
end


function testViterbi(mlp, criterion, dataset)
   local n = 0
   for t=1,dataset:size() do
      n = n + #(dataset[t][1])
   end

   local predictionSentence = {}
   local predictions = torch.Tensor(n)
   print('[we have to test ' .. n .. ' windows]')
   n = 0

   for t=1,dataset:size() do
      local example = dataset[t]
      local path = criterion.viterbi:forward(mlp:forward(example[1]), example[2])
      local label = example[2]

      predictionSentence[t] = torch.Tensor(path)
      ----print(path:size(1))
      for i=1,path:size(1) do
         if not label[i] then
            error('bizarre bug')
         end
         n = n + 1
         predictions[n] = path[i]
      end
   end
   print('[we did test on ' .. n .. ' windows]')

   return predictions, predictionSentence 
end




function loadHash(params, hashhome)
   local hash = {}
   print('[loading tag hash for task : ]' .. params.task .. ' ; ')
   
   ---- cb513ssp.cv7, dssalltaskdssp, dssalltaskssp, dssalltasksar, dssalltasksaa, coilcoil.cv5, dna-binding.cv3, ppi.cv5, sp.cv10, tm.cv10
   if params.task:sub(1,8) == 'cb513ssp' or params.task == 'dssalltaskssp' then
         for line in  io.lines(hashhome.. '/ssp-tag.lst') do
            table.insert(hash, line)
         end   
   elseif params.task == 'dssalltaskdssp' then
      for line in io.lines(hashhome.. '/dssp-tag.lst') do
         table.insert(hash, line)
      end
   elseif params.task == 'dssalltasksaa' then
         for line in io.lines(hashhome .. '/sa-absolute-tag.lst') do
            table.insert(hash, line)
         end
   elseif params.task == 'dssalltasksar' then
         for line in io.lines(hashhome .. '/sa-relative-tag.lst') do
            table.insert(hash, line)
         end      
   elseif params.task:sub(1,8) == 'coilcoil' then
         for line in io.lines(hashhome .. '/coilcoil-tag.lst') do
            table.insert(hash, line)
      end      
   elseif params.task:sub(1,11) == 'dna-binding' then
         for line in io.lines(hashhome .. '/dna-binding-tag.lst') do
            table.insert(hash, line)
      end        
   elseif params.task:sub(1,3) == 'ppi' or params.task:sub(1,4) == 'tppi'  then
         for line in io.lines(hashhome .. '/ppi-tag.lst') do
            table.insert(hash, line)
      end              
   elseif params.task:sub(1,2) == 'sp' then
         for line in io.lines(hashhome .. '/sp-tag.lst') do
            table.insert(hash, line)
      end                    
   elseif params.task:sub(1,2) == 'tm' then
         for line in io.lines(hashhome .. '/tm-tag.lst') do
            table.insert(hash, line)
      end                    
   else
      error('invalid task')
   end
   print('[Task tag hash loaded]')
   return hash
end



function save_predict(predictions, predictionSentence, predfileN, dicthome, params, testset, origFastaData)

   hashes = hashes or loadHash(params, dicthome) 
   local numSeq = table.getn(predictionSentence)
   print('[number of available AA predictions: ] ' .. predictions:size(1))
   print('[number of predicted protein sequences: ] ' .. numSeq)
 
   local n = 0
   local  f = io.open(predfileN, "w")      
   for t=1, numSeq do
      f:write(">".. origFastaData[t].comment .. "\n")
      local path = predictionSentence[t]
      ----print(path:size(1))
      for i=1,path:size(1) do
         local hpred = path[i]
	 if hashes[hpred] then 
	    f:write(hashes[hpred])
	    n = n + 1
	 end
      end
      f:write("\n")
   end
   print('[we save prediction for ' .. n .. ' windows]')
   f:close()
end



function save_predict2(predictions, predfileN, dicthome, params, testset, origFastaData)

   hashes = hashes or loadHash(params, dicthome) 
   print('[number of available AA predictions: ] ' .. predictions:size(1))
 
   local n = 0
   local numSeq = 1; 
   local midPos = (params.niw + 1) / 2
   local  f = io.open(predfileN, "w")            
   for i=1, predictions:size(1)  do
      if (i ==1) then
	 f:write(">".. origFastaData[numSeq].comment .. "\n")
	 numSeq = numSeq + 1
      end

      local hpred = predictions[i]
      if hashes[hpred] then 
	 f:write(hashes[hpred])
	 n = n + 1
      end

      if params.psi and params.psi >0 and (i ~=  predictions:size(1)) then 
	 if (testset[i][1][1][midPos + 1] <= -2)  then  --end of a sentence; this only works on PSI features / not applicable to AA features 
	    print(i)
	    f:write("\n")
	    f:write(">".. origFastaData[numSeq].comment .. "\n")
	    numSeq = numSeq + 1
	 end
      elseif params.a1sz and params.a1sz >0 and (i ~=  predictions:size(1)) then
	 if  (testset[i][1][1][midPos + 1] ==1) and (i ~=  predictions:size(1)) then  --end of a sentence; this only works on AA features / not applicable to PSI features 
	    print(i)
	    f:write("\n")
	    f:write(">".. origFastaData[numSeq].comment .. "\n")
	    numSeq = numSeq + 1
	 end
      end      
   end
   f:write("\n")
   f:close()
   print('[we save prediction for ' .. n .. ' windows]')
   numSeq = numSeq -1
   print('[we save predicted protein sequences: ] ' .. numSeq)

end