require 'torch'

local CmdLine = torch.class('torch.CmdLine')

local function strip(str)
   return string.match(str, '%-*(.*)')
end

local function pad(str, sz)
   return str .. string.rep(' ', sz-#str)
end

function CmdLine:__readArgument__(params, arg, i, nArgument)
   local argument = self.arguments[nArgument]
   local value = arg[i]

   if nArgument > #self.arguments then
      error('do not see what you want to do with ' .. value)
   end
   if argument.type and type(value) ~= argument.type then
      error('invalid argument type for argument ' .. argument.key .. ' (should be ' .. argument.type .. ')')
   end
   params[strip(argument.key)] = value
   return 1
end

function CmdLine:__readOption__(params, arg, i)
   local key = arg[i]
   local option = self.options[key]
   if not option then
      error('unknown option ' .. key)
   end

   if option.type and option.type == 'boolean' then
      params[strip(key)] = not option.default
      return 1
   else
      local value = arg[i+1]
      if not value then
         error('missing argument for option ' .. key)
      end
      if not option.type or option.type == 'string' then
      elseif option.type == 'number' then
         value = tonumber(value)
      else
         error('unknown required option type ' .. option.type)
      end
      if not value then
         error('invalid type for option ' .. key .. ' (should be ' .. option.type .. ')')
      end
      params[strip(key)] = value
      return 2
   end
end

function CmdLine:__init()
   self.options = {}
   self.arguments = {}
   self.helplines = {}
end

function CmdLine:argument(key, help, _type_)
   table.insert(self.arguments, {key=key, help=help, type=_type_})
   table.insert(self.helplines, self.arguments[#self.arguments])
end

function CmdLine:option(key, default, help, _type_)
   if default == nil then
      error('option ' .. key .. ' has no default value')
   end
   _type_ = _type_ or type(default)
   if type(default) ~= _type_ then
      error('option ' .. key .. ' has wrong default type value')
   end
   self.options[key] = {key=key, default=default, help=help, type=_type_}
   table.insert(self.helplines, self.options[key])
end

function CmdLine:default()
   local params = {}
   for option,v in pairs(self.options) do
      params[strip(option)] = v.default
   end
   return params
end

function CmdLine:parse(arg)
   local i = 1
   local params = self:default()

   local nArgument = 0

   while i <= #arg do
      if arg[i] == '-help' or arg[i] == '-h' or arg[i] == '--help' then
         self:help(arg)
         os.exit(0)
      end

      if self.options[arg[i]] then
         i = i + self:__readOption__(params, arg, i)
      else
         nArgument = nArgument + 1
         i = i + self:__readArgument__(params, arg, i, nArgument)
      end
   end

   if nArgument ~= #self.arguments then
      error('not enough arguments')
   end

   return params
end

function CmdLine:string(str, params)
   str = '-' .. str .. '-'
   for k,v in pairs(params) do
      if type(v) == 'boolean' then
         if v then
            v = 't'
         else
            v = 'f'
         end
      end
      str = string.gsub(str, '(%p)%$' .. strip(k) .. '(%p)', '%1' .. strip(k) .. '=' .. v .. '%2')
   end
   str = string.match(str, '%-(.*)%-')
   return str
end

function CmdLine:log(file, params)   
   ----print(file)
   local f = io.open(file, 'w')
   local oprint = print
   function print(...)
      local n = select("#", ...)
      oprint(...)
      for i=1,n do
         f:write(tostring(select(i, ...)))
         if i ~= n then
            f:write(' ')
         else
            f:write('\n')
         end
      end
      f:flush()
   end
   print('[program started on ' .. os.date() .. ']')
   print('[command line arguments]')
   if params then
      for k,v in pairs(params) do
         print(k,v)
      end
   end
   print('[----------------------]')   
end

function CmdLine:text(txt)
   txt = txt or ''
   assert(type(txt) == 'string')
   table.insert(self.helplines, txt)
end

function CmdLine:help(arg)
   io.write('Usage: ')
   if arg then io.write(arg[0] .. ' ') end
   io.write('[options] ')
   for i=1,#self.arguments do
      io.write('<' .. strip(self.arguments[i].key) .. '>')
   end
   io.write('\n')

   -- first pass to compute max length
   local optsz = 0
   for _,option in ipairs(self.helplines) do
      if type(option) == 'table' then
         if option.default ~= nil then -- it is an option
            if #option.key > optsz then
               optsz = #option.key
            end
         else -- it is an argument
            if #strip(option.key)+2 > optsz then
               optsz = #strip(option.key)+2
            end
         end
      end
   end

   -- second pass to print
   for _,option in ipairs(self.helplines) do
      if type(option) == 'table' then
         io.write('  ')
         if option.default ~= nil then -- it is an option
            io.write(pad(option.key, optsz))
            if option.help then io.write(' ' .. option.help) end
            io.write(' [' .. tostring(option.default) .. ']')
         else -- it is an argument
            io.write(pad('<' .. strip(option.key) .. '>', optsz))
            if option.help then io.write(' ' .. option.help) end
         end
      else
         io.write(option) -- just some additional help
      end
      io.write('\n')
   end
end
