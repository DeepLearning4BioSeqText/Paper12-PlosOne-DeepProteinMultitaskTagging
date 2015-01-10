require "torch"

iolearn = {}

function iolearn.load(name, maxLoad)
   local f = torch.DiskFile(name, "r")
   local maxPosition = 100
   local positions = torch.IntStorage(100)
   local currentPosition = 1
   local n = 0
   local maxValue = -math.huge
   local minValue = math.huge
   print('# Reading ' .. name)
   f:quiet()
   while true do
      local size = f:readInt()
      if f:hasError() then break end
      local stuff = torch.Tensor(f:readDouble(size), 1, size, -1)
      maxValue = math.max(maxValue, stuff:max())
      minValue = math.min(minValue, stuff:min())
      if f:hasError() then break end

      n = n + 1
      if n > maxPosition then
         maxPosition = math.ceil(maxPosition*1.5)
         positions:resize(maxPosition, true)
      end
      positions[n] = currentPosition
      currentPosition = currentPosition + size + 1

      if maxLoad and maxLoad > 0 and n == maxLoad then break end

   end
   positions:resize(n, true)
   f:close()
   f = torch.DiskFile(name, "r")
   local storage = f:readDouble(currentPosition-1)
   f:close()

   local data = {min=minValue, max=maxValue}
   function data:size()
      return n
   end

   setmetatable(data, {__index = function(self, index)
                                    local position = positions[index]
                                    return torch.Tensor(storage, position+1, storage[position])
                                 end})
   return data
end

function iolearn.windows(data, windowSize, padding)
   padding = padding or 1
   local size = 0
   for i=1,data:size() do
      size = size + data[i]:size(1)
   end

   local winidx2dataidx = torch.IntTensor(size)
   local winidx2elemidx = torch.IntTensor(size)
   local offset = 1
   for i=1,data:size() do
      winidx2dataidx:narrow(1, offset, data[i]:size(1)):fill(i)

      local elemidx = 0
      winidx2elemidx:narrow(1, offset, data[i]:size(1)):apply(function()
                                                                 elemidx = elemidx + 1
                                                                 return elemidx
                                                              end)
      offset = offset + data[i]:size(1)
   end

   local windows = {}
   function windows:size()
      return size
   end

   setmetatable(windows, {__index = function(self, index)
                                       local dataidx = winidx2dataidx[index]
                                       local elemidx = winidx2elemidx[index]
                                       local fDx = math.max(0, (windowSize-1)/2 - elemidx + 1)
                                       local lDx = math.max(0, (windowSize-1)/2 + elemidx - data[dataidx]:size(1))
                                       local window = torch.Tensor(windowSize):fill(padding)                                       
                                       local buffer = data[dataidx]:narrow(1, elemidx - (windowSize-1)/2 + fDx, windowSize-fDx-lDx)
                                       window:narrow(1, 1+fDx, windowSize-fDx-lDx):copy(buffer)
                                       return window
                                    end})

   return windows
end

function iolearn.padding(data, paddingSize, paddingValue)
   local padding = {}
   function padding:size()
      return data:size()
   end

   setmetatable(padding, {__index = function(self, index)
                                       local size = data[index]:size(1)
                                       local tensor = torch.Tensor(size+2*paddingSize):fill(paddingValue)
                                       tensor:narrow(1, paddingSize+1, size):copy(data[index])
                                       return tensor
                                    end})
   return padding
end

function iolearn.iterator(data)
   local size = 0
   for i=1,data:size() do
      size = size + data[i]:size(1)
   end
   
   local winidx2dataidx = torch.IntTensor(size)
   local winidx2elemidx = torch.IntTensor(size)
   local offset = 1
   for i=1,data:size() do
      winidx2dataidx:narrow(1, offset, data[i]:size(1)):fill(i)

      local elemidx = 0
      winidx2elemidx:narrow(1, offset, data[i]:size(1)):apply(function()
                                                                 elemidx = elemidx + 1
                                                                 return elemidx
                                                              end)
      offset = offset + data[i]:size(1)
   end

   local iterator = {}
   function iterator:size()
      return size
   end

   setmetatable(iterator, {__index = function(self, index)
                                        return {winidx2dataidx[index], winidx2elemidx[index]}
                                     end})

   return iterator
end

function iolearn.distance(maxRange, maxSize)
   maxSize = maxSize or 1001 -- maximum sentence size*2+1
   local distidx = torch.Tensor(maxSize)

   local halfMaxSize = (maxSize+1)/2
   if maxRange <= 2 then
      distidx:fill(1)
      distidx[halfMaxSize] = 2
   else
      local halfDistRange = (maxRange-1)/2
      local c = 1
      for i=1,halfMaxSize-halfDistRange-1 do
         distidx[i] = c
      end
      for i= halfMaxSize-halfDistRange, halfMaxSize+halfDistRange do
         distidx[i] = c
         c = c + 1
      end
      c = c - 1
      for i = halfMaxSize+halfDistRange+1, maxSize do
         distidx[i] = c
      end
   end

   local distance = {}
   setmetatable(distance, {__call = function(self, idx, size)
                                       return distidx:narrow(1, halfMaxSize-idx+1, size)
                                    end})

   return distance
end
