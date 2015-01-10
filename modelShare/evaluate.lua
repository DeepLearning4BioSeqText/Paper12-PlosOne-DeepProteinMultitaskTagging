require "torch"

-- Split text into a list consisting of the strings in text,
function strip(str)
  local t = {}
  local function helper(word) table.insert(t, word) return "" end
  if not str:gsub("%a", helper):find"%S" then return t end
end


function evaluate(labelf, predictf)
   print('[evaluating accuracy score] ')
   print('[test prediction file ] ' .. predictf)
   print('[test label file ] ' .. labelf)

   local fl=io.open(labelf,"r")
   local fp=io.open(predictf,"r")

   local AAerror = 0; 
   local Proterror = 0; 
   local AAtotal = 0; 
   local Prottotal = 0; 

   while true do
      local line=fl:read("*line")
      local linep=fp:read("*line")

      if line==nil then break; end
      if linep==nil then break; end

      if line:sub(1,1)==">" and linep:sub(1,1)==">" then  
         Prottotal = Prottotal + 1; 
      elseif line:sub(1,1)==">" and linep:sub(1,1) ~=">" then
	 print('Label file and predict file format mismatched !')
      elseif line:sub(1,1)~=">" and linep:sub(1,1) ==">" then  
	 print('Label file and predict file format mismatched !')
      else
         if string.len(line)>0 and string.len(line) == string.len(linep) then
	    AAtotal = AAtotal +  string.len(linep)
	    local ary = strip( line)
	    local aryP = strip( linep)
	    local curMatch =  string.len(linep)
 
	    for key,value in ipairs(ary) do
	       ----- print(key, value, aryP[key])   --- for debugging 
	       if value ~= aryP[key] then
		  AAerror = AAerror + 1
		  curMatch = curMatch - 1
	       end
	    end 
	    if curMatch ~= string.len(line) then 
	       Proterror = Proterror + 1
	    end
	 else
	    print('[Error ('.. Prottotal ..')] Label file and predict file line length mismatched !')
	 end
      end
   end

   local epsino = 0.00000000001
   local AAerrorRate = AAerror / (AAtotal + epsino)
   local ProterrorRate = Proterror / (Prottotal + epsino)

   print("[AA level error rate] : " .. tostring(AAerrorRate))
   print("[AA total Num] : " .. tostring(AAtotal))
   ----print("[Protein level error rate] : " .. tostring(ProterrorRate))
   print("[Protein total Num] : " .. tostring(Prottotal))

   fl:close()
   fp:close()
end


--- for debugging 
---evaluate(arg[1], arg[2])