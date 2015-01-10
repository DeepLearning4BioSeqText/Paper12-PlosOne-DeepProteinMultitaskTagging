require "posix"

local hashDict={};
 
function hashDict.create()
	 local self={}; 
	 local meta={}; setmetatable(self,meta);
	 local function client_index(table,field)
		return hashDict[field];
	 end
	 meta.__index = client_index;
 	 self.data={}; 
	 self.data._size_=0;
	 self.field='orig'
	 return self;
end

function hashDict.select(self,new_field)
	self.field=new_field;
end

function hashDict.load_list(self,fname,hname,max)
	
    n=hname:sub(1,hname:len()-4)
    local data=self.data;
    if data[n]~=nil then 
	data["_size_"] = data["_size_"] - data[n]["_size_"];
    end
    data[n] = {};
    local count = 0
    local file = io.open(fname,'r')
    local tab = data[n]

    for word in file:lines() do
       count = count + 1
       if count % 100000 == 0 then io.stderr:write('.') end
       tab[count] = word
       tab[word] = count
       if max~=nil and count>=max then break; end
    end

    file:close()
    tab["_size_"] = count 
    data["_size_"] = data["_size_"] + count;

    print("[loaded " .. n .. " : " .. count .. " words]")

end

function hashDict.load(self,fname,max) -- load database
   
	-- if load directory, load all, otherwise load file
	-- count sizes as we load..
	local t=posix.stat(fname);  
	print(fname)
	if t.type=="regular" then
	        local hname
 	        local cutpath=0
	        for i=1,fname:len() do
		   local p=fname:find('/',i)
		   if p~=nil then cutpath=p; end
		end
		hname=fname:sub(cutpath+1,fname:len())
		self:load_list(fname,hname,max);
		local h=hname:sub(1,hname:len()-4)
		self:select(h)
	else -- load directory
		print("loading directory of files..",fname)
		local dir=posix.dir(fname);
		for i,k in pairs(dir) do
			local f=fname .. "/" .. k;
			if posix.stat(f).type=="regular" then
				self:load_list(f,k,max)
			end
		end
	end
end

function hashDict.save_list(self,fname,t)
	print("[saving " .. fname  .. "]")
	f=io.open(fname,'w')
	for i=1,t._size_ do
		f:write(t[i] .. "\n");
	end	
	f:close()
end

function hashDict.save(self,fname,field) -- load database
	if field~=nil then
	 self:save_list(fname,self.data[field]);
	else
	 posix.mkdir(fname);
	 for i,k in pairs(self.data) do
		if type(k)=="table" then
			self:save_list(fname .. "/" .. i .. ".lst",k)
		end
	 end			
        end
end

function hashDict.getIndex(self,key,create_if_not_exist)
	if self.data[self.field]~=nil and
	   self.data[self.field][key]~=nil then 
		return self.data[self.field][key],false;  
	end
	local s;  local created=false;

	if create_if_not_exist then
	        created=true;
		local data=self.data; local field=self.field;
		if data[field]==nil then
			data[field]={};
			data[field]._size_=0;
		end
		data[field]._size_=data[field]._size_+1;
		data._size_ = data._size_+1;
		s=data[field]._size_;
		data[field][key]=s; -- index
		data[field][s]=key  -- reverse index
	end
	return s,created;
end


function hashDict.getReverseIndex(self,key)
	if self.data[self.field]~=nil and
	   self.data[self.field][key]~=nil then 
		return self.data[self.field][key];  
	end
	local s=nil; 
	if self.port~=nil then -- ** network version checks online
		s=self.client:get_key(self.field,key)
	end
	return s;
end

hashdict=hashDict;
 