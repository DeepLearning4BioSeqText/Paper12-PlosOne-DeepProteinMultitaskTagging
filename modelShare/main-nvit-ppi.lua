--------------------------------------------------------------------------------------------------
----  Please revise three variables before running ! 
----  1) "dicthome" in "main.lua" to your dictionary location 
----  2) "binDir"  in "getPsiBlastNR.py" to your own BLAST local location  
----  3) "nrdb" in "getPsiBlastNR.py" to your own BLAST db local location  
----  4) "psipath" in "main.lua" to your local directory including psiblast profile features   
--------------------------------------------------------------------------------------------------

--- ~/torchnew/build/bin/lua main-nvit-ppi.lua ppi ./test/fasta/input.tppi.seq.faa ~/MultSeq/backGoodModels/models4shareOnline/TPPI-nonVit/tppi.psionly.model.asc.noviterbi ./test/ ./test/fasta/input.tppi.lab.faa 1

require "torch"
require "nn"

dofile('inc_ioformat.lua')
dofile('CmdLine.lua')
dofile('inc_dataset.lua')
dofile('inc_test.lua')
dofile('evaluate.lua')

--- Processing Input Args
cmd = torch.CmdLine()
cmd:text()
cmd:text('Predict AA labels for certain task on input fasta file')
cmd:text()
cmd:text('Arguments:')
cmd:argument('-task', 'task: ppi')
cmd:argument('-testf', 'test fasta file to predict on')
cmd:argument('-umodel', 'a previously trained network model')
cmd:argument('-outdir', 'subdirectory to save the predictions in')
cmd:argument('-labelf', 'test fasta file to evaluate accuracy')
cmd:argument('-psiExistFlag', 'psiExistFlag: 1 indicating psiBlast features exist, with system-defined dir  ; 2  indicating psiBlast features exist, with user-defined dir in psipath; 0 indictating not existing ')
cmd:text()

params = cmd:parse(arg)
params.niw = 13
params.a1sz = 0
params.psi = 2


rundir = params.outdir .."/".. params.task..'/'
print("[Prediction Output Dir: ] " .. rundir)
os.execute('mkdir -p ' .. rundir)
cmd:log(rundir .. '/log', params)


--- Reading the old model 
print('[ Reading from old good model ] '.. params.umodel )
local f = torch.DiskFile( params.umodel )
local mlp = f:readObject():get(1)
f:close()
print('[ok]')


--- Preprocessing the input fasta format 
--- PLEASE revise "dicthome" to your dictionary location 
--- PLEASE revise "psipath" to your psifeature files local location if existing 
--- PLEASE ALSO REVISE "binDir" and "nrdb" in "getPsiBlastNR.py" to your own BLAST local location  
local dicthome = "./hash/"
---local psipath = "/net/multiSequence/psiblastProfilesDSSP-NRF/"
local psipath = "./test/psifile/"
local fastaOrigData = io_format(params.testf, rundir, dicthome, tonumber(params.psiExistFlag), psipath)


---- Load test data 
local testDatasetOrig = createDataset(rundir, params ) 
print("[ Creating Testing set - Num. of AAs ] ".. testDatasetOrig:size())  


--- We predict on test on non-viterbi models 
local predictions, scores = test(mlp, testDatasetOrig)
local fpredict = rundir .. '/test-predict'
save_predict2(predictions, fpredict, dicthome, params, testDatasetOrig, fastaOrigData)


--- We check accuracy if label file exists 
evaluate(params.labelf, fpredict)