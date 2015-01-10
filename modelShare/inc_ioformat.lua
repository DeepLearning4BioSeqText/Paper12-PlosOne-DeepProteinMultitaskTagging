
-- creates fasta file and generate feature
require "torch"
dofile "hashdict.lua"

psiUsedNum = 20

---load in the sequence 
function load_fasta(fname)
   local data={};
   local t={}
   local f=io.open(fname,"r")
   while true do

      local line=f:read("*line")
      if line==nil or line:sub(1,1)==">" then -- add old example to list
         if t.data ~=nil then data[#data+1]=t; end
      end
      if line==nil then break; end

      if line:sub(1,1)==">" then -- add old example to list
         --start new example
         t={}
         t.comment = line;
      else
         if string.len(line)>0 then
            if t.data==nil then
               t.data=line;
            else
               t.data = t.data .. line
            end
         end
      end

   end
   return data;
end




--equal to perl's split function, "inSplitPattern" as the split delimiter
--if several==true, then several consecutive delimiters will be cosidered as one
--returns an array of pieces of the string.
function strsplit( inSplitPattern, stringIn , several)  
   local outResults = {}
   local theStart = 1
   if string.find("", inSplitPattern, 1) then
      error("Split delimiter matches empty string!")
   end
   local theSplitStart, theSplitEnd = string.find( stringIn, inSplitPattern, theStart )
   while theSplitStart do
      table.insert( outResults, string.sub( stringIn, theStart, theSplitStart-1 ) )
      theStart = theSplitEnd + 1
      theSplitStart, theSplitEnd = string.find( stringIn, inSplitPattern, theStart )
   end
   table.insert( outResults, string.sub( stringIn, theStart ) )
   if several==true then
      local outResults2={}; 
      for i=1,#outResults do 
         if not string.find("",outResults[i],1) then 
            table.insert(outResults2,outResults[i]) 
         end
      end
      return outResults2 
   else 
      return outResults
   end
end



--to check how many data lines the mtx-file has 
function checkFileSiz(filename, emtpyheaderlines)
   local numExample = 0
   fdpt = assert(io.open(filename,'r'))
   if emtpyheaderlines > 0 then
     for i=1, emtpyheaderlines do
          local line = fdpt:read() -- emtpy line
     end
   end
   while (true) do
      local line = fdpt:read()
      if string.len(line)==0 
      then break end
      numExample = numExample + 1
   end
   fdpt:close();
   print("[ Input mtx file Num. lines: ]".. numExample .." data lines.")
   return numExample
end



 function exists(filename)
          local file = io.open(filename)
          if file then
            io.close(file)
            return true
          else
            return false
          end
 end



---load in the sequence psi-profile 

function load_psiblast(seqname,path)
   local data={};
   local t={}
   local filename = path.."/"..seqname:sub(1,1).."/"..seqname:sub(2,2).."/"..seqname..".mtx"
   if exists(filename) then
      ---print(filename)
      local Nlines = checkFileSiz(filename, 3)
      local f=io.open(filename,"r")
      local line=f:read("*line")
      local line=f:read("*line")
      local line=f:read("*line")
      local header=strsplit(" ",line,true)
      if #header ~= 40 then
         error("Header row should have 40 labels!") 
      end
      local z = torch.Tensor(psiUsedNum, Nlines):fill(0)
      local k = 0;
      while true do
         local line=f:read("*line")
         if line==nil or string.len(line)==0 then 
            break; 
         else
            k=k+1;
            local row = strsplit(" ",line,true)
            if #row ~= 44 then
               error("Data row should have 44 columns!") 
            end
            if k ~= tonumber(row[1]) then
               print(k)
               print(row[1])
               error("Row numbering somehow mixed up!")
            end
            for j= 3, 22 do   --- the first 20 values in each line 
               z[j-2][k]=row[j]
            end
         end
      end
      f:close();
      return z;
   else
      print('Not exist:'..filename)
   end
      return nil; 
end




function loadDict(dicthome, dtype)
   -- home directory with word/AA dictionaries
   if (dict_words == nil ) then
      dict_words = hashdict.create()
      dict_words:load(dicthome .. "aa1.lst")
   end
end


---- map/extract features
function  outInd(db, word, outfh) 
   local res,isnew = db:getIndex(word,true)
   if isnew then
      print("::::: Entry is not in the dictionary : '" .. db.field .. "' .. '" .. word .. "'")
   end
   outfh:write(res.." ")
   return res,isnew
end



function parse2Ind(proteinS, choice, outfh )
   if proteinS == nil then
      print("EMPTY PROT \n")
      return 
   end    
   outfh:write(proteinS:len().."\n")  
      
   for word in proteinS:gmatch('%a') do   -----  
      outInd(dict_words, word, outfh) 
   end
   outfh:write("\n")
   return 
end





---convert both psilast and the string aa
function convertPsiBlastNew( outhome, data, pathPsiblastprofiles, psifileSuf)

   local f_aa = {};
   local aaLimit = 1;  
   for i=1, aaLimit do
      f_aa[i] = io.output(outhome .. "aa" .. tostring(i) .. ".dat", "w")
   end

   local f_psi = {}
   for i = 1 , psiUsedNum do 
      f_psi[i] = io.output(outhome .. "psi" .. tostring(i) .. ".dat." ..psifileSuf, "w")      
   end   

   for i=1,#data do
         local CurWfea = {}; 
         for j=1, aaLimit do
            CurWfea[j] = parse2Ind(data[i].data, j, f_aa[j] )            
         end
         
        --print(string.sub(data[i].comment,2))
         comment = string.sub(data[i].comment,2)
         psi = load_psiblast(string.sub(data[i].comment,2), pathPsiblastprofiles)
         if psi == nil then 
            psi = torch.Tensor(psiUsedNum, string.len(data[i].data)):fill(-100)
         end 
         for psiInd = 1, psiUsedNum do 
            f_psi[psiInd]:write(string.len(data[i].data).."\n")  
         end
         local curPsiAry = torch.Tensor(psiUsedNum, psi:size(2))
         for curpsi = 1, psi:size(2) do
            for psiInd = 1, psiUsedNum do
               f_psi[psiInd]:write(tostring(psi[psiInd][curpsi]).." ")
               curPsiAry[psiInd][curpsi] =  tostring(psi[psiInd][curpsi])
            end
         end
         for psiInd = 1, psiUsedNum do 
            f_psi[psiInd]:write("\n")  
         end        
   end

   for i=1, aaLimit do
      f_aa[i]:close()
   end
   for i = 1 , psiUsedNum do 
      f_psi[i]:close()   
   end 
   print("[Num of Processed Sequences: ]" .. #data)
end





function io_format(inputfastaf, outhome, dicthome, psiExistFlag, psipath)

   print(string.format("[ Hash Dict ]: ".. dicthome .. ";" ))
   loadDict(dicthome)

   local dfx=load_fasta(inputfastaf)
   print(string.format("[ Extracted Num sequences:  ] " .. #dfx ))

   print("[ Output Dir: ] ".. outhome .. " ;")
   os.execute('mkdir -p ' .. outhome)

   local psiType = "NRF"
   ----local psifileSuf =  outhome.."/psifile-"..psiType.."/"; 
   local psifileSuf =  psipath.."/"..psiType.."/";
   if psiExistFlag < 2 then 
      os.execute('mkdir -p ' ..psifileSuf)
   end

   ---- psiExistFlag: 1 indicating psiBlast features exist, with system-defined dir  ; 2  indicating psiBlast features exist, with user-defined dir in psipath; 0 indictating not existing
   if psiExistFlag == 0 then 
      local psiCommand = 'python getPsiBlastNR.py   '.. inputfastaf .. '  '..psifileSuf
      local retcode = os.execute(psiCommand)
      if tonumber(retcode) ~= 0 then
	 io.stderr:write("[ ** ERROR: PSIBlast ]\n")
	 return 1
      end
   elseif psiExistFlag == 2 then 
      psifileSuf = psipath
   else 
      psifileSuf = psifileSuf 
   end
   print("[ PSIBLAST output Dir: ]  "..psifileSuf.. "  ;")
   convertPsiBlastNew(outhome, dfx, psifileSuf, psiType)
   return dfx
end

--- To Test: 
--- io_format(arg[1], arg[2], arg[3])