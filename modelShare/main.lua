--------------------------------------------------------------------------------------------------
----  Please revise three variables before running ! 
----  1) "dicthome" in "main.lua" to your dictionary location 
----  2) "binDir"  in "getPsiBlastNR.py" to your own BLAST local location  
----  3) "nrdb" in "getPsiBlastNR.py" to your own BLAST db local location  
----  4) "psipath" in "main.lua" to your local directory including psiblast profile features   
--------------------------------------------------------------------------------------------------

require "torch"
require "nn"
require "viterbi"

dofile('inc_ioformat.lua')
dofile('CmdLine.lua')
dofile('inc_dataset.lua')
dofile('inc_test.lua')
dofile('evaluate.lua')
dofile('SequenceWrapper.lua')
dofile('AlphaCriterion2.lua')

--- Processing Input Args
cmd = torch.CmdLine()
cmd:text()
cmd:text('GOAL: Predict AA labels for certain task on input fasta file')
cmd:text()
cmd:text('Arguments:')
cmd:argument('-task', 'task: cb513ssp, dssalltaskdssp, dssalltaskssp, dssalltasksar, dssalltasksaa, coilcoil, dna-binding, ppi, sp, tm')
cmd:argument('-testf', 'test fasta file to predict on')
cmd:argument('-umodel', 'a previously trained network model')
cmd:argument('-outdir', 'subdirectory to save the predictions in')
cmd:argument('-labelf', 'test fasta file to evaluate accuracy')
cmd:argument('-psiExistFlag', 'psiExistFlag: 1 indicating psiBlast features exist, with system-defined dir  ; 2  indicating psiBlast features exist, with user-defined dir in psipath; 0 indictating not existing ')
cmd:text()

params = cmd:parse(arg)
params.niw = 13
params.a1sz = 15
params.psi = 2

rundir = params.outdir .."/".. params.task..'/'
print("[Prediction Output Dir: ] " .. rundir)
os.execute('mkdir -p ' .. rundir)
cmd:log(rundir .. '/log', params)


--- Reading the old model 
print('[ Reading from old good model ] '.. params.umodel )
local f = torch.DiskFile( params.umodel )
vmlp = f:readObject()
criterion = f:readObject()
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
local testDataset = createSentenceDataset(testDatasetOrig)
testDataset = createWeightedDataset(testDataset, 'word')
local maxSentence = 0
for i=1,testDataset:size() do
   if #(testDataset[i][1]) > maxSentence then
      maxSentence = #(testDataset[i][1])
   end
end
print("[ Creating testing set viterbi version - Num of examples: ] ".. testDataset:size())        


--- We predict on test 
local predictions, predictionSentence = testViterbi(vmlp, criterion, testDataset)
local fpredict = rundir .. '/test-predict'
save_predict(predictions, predictionSentence, fpredict, dicthome, params, testDataset, fastaOrigData)


--- We check accuracy if label file exists 
evaluate(params.labelf, fpredict)
