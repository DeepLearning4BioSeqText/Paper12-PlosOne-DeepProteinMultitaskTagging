# Paper12-PlosOne-DeepProteinMultitaskTagging

Y. Qi, M. Osh, J. Weston, W. Noble (2012) 
A unified multitask architecture for predicting local protein properties, PLoS ONE (March 2012) (URL), 



Summary: Deep neural network architecture + Protein Sequence Labeling

(bibTex), 
@article{qi12plosone,
    author = {Qi, , Yanjun AND Oja, , Merja AND Weston, , Jason AND Noble, , William Stafford},
    journal = {PLoS ONE},
    publisher = {Public Library of Science},
    title = {A Unified Multitask Architecture for Predicting Local Protein Properties},
    year = {2012},
    month = {03},
    volume = {7},
    url = {http://dx.doi.org/10.1371%2Fjournal.pone.0032235},
    pages = {e32235},
    abstract = {<p>A variety of functionally important protein properties, such as secondary structure, transmembrane topology and solvent accessibility, can be encoded as a labeling of amino acids. Indeed, the prediction of such properties from the primary amino acid sequence is one of the core projects of computational biology. Accordingly, a panoply of approaches have been developed for predicting such properties; however, most such approaches focus on solving a single task at a time. Motivated by recent, successful work in natural language processing, we propose to use <italic>multitask learning</italic> to train a single, joint model that exploits the dependencies among these various labeling tasks. We describe a deep neural network architecture that, given a protein sequence, outputs a host of predicted local properties, including secondary structure, solvent accessibility, transmembrane topology, signal peptides and DNA-binding residues. The network is trained jointly on all these tasks in a supervised fashion, augmented with a novel form of semi-supervised learning in which the model is trained to distinguish between local patterns from natural and synthetic protein sequences. The task-independent architecture of the network obviates the need for task-specific feature engineering. We demonstrate that, for all of the tasks that we considered, our approach leads to statistically significant improvements in performance, relative to a single task neural network approach, and that the resulting model achieves state-of-the-art performance.</p>},
    number = {3},
    doi = {10.1371/journal.pone.0032235}
}        



<BR>


Paper URL:  http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0032235 

<BR>


(SupplementWeb), all trained deep models have been shared @ 
http://noble.gs.washington.edu/proj/multitask/models/



--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
RELATED SOFTWARE INSTALLATION: 

---- (1) install BLAST from ncbi (NR filtered db)   
+ http://www.ncbi.nlm.nih.gov/staff/tao/URLAPI/blastpgp.html#1.1
+ setup using instrusction from : 
  http://www.ncbi.nlm.nih.gov/staff/tao/URLAPI/unix_setup.html
  ftp://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/  

+ BLAST database is a key component of any BLAST search. 
  Here we need to first download the NR database ==>
  ftp ftp.ncbi.nlm.nih.gov  using "anonymous" ,  
  then " cd blast/db ", 
  then "get nr.*tar.gz" , 
  then "tar zxvpf nr.*tar.gz"
  ==> nr.*tar.gz (non-redundant protein sequence database with entries from GenPept, Swissprot, PIR, PDF, PDB, and NCBI RefSeq )

+ Configuration:
  http://www.ncbi.nlm.nih.gov/staff/tao/URLAPI/blastpgp.html#3.2

+ We can further filter nr database to nrf database using softwares provided by PSIPred
  - This step is optional !
  - Software PSIPred : http://bioinfadmin.cs.ucl.ac.uk/downloads/psipred/README
  - A program called "pfilt" is included which will filter FASTA files before
    using the formatdb command to generate the encoded BLAST data bank files.
  - For example:
      pfilt nr.fasta > nrfilt
      formatdb -t nrfilt -i nrfilt
      cp nrfilt.p?? $BLASTDB

  - (The above command assumes you have already set the BLASTDB environment variable 
     to the directory where you usually keep your BLAST data banks)
  
--------------------------------------------------------------------------------------------------
---- (2) install TORCH5   http://torch5.sourceforge.net/

+ Please following step by step using the page: 
  http://torch5.sourceforge.net/manual/Install-2.html

--------------------------------------------------------------------------------------------------
---- (3) install viterbi package in your torch package 

+ After the installation of TORCH5, copy the whole "viterbi" package directory under "your-path-to-torch-sources/dev/"
+ Then recompile and run commands ==>
  cd your-path-to-torch-sources/build
  cmake .. -DCMAKE_INSTALL_PREFIX=/my/install/path
  make
  make install

--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
----  Please revise three variables before running ! 
----  1) "binDir"  in "getPsiBlastNR.py" to your own BLAST local location  
----  2) "nrdb" in "getPsiBlastNR.py" to your own BLAST db local location  
----  3) "psipath" in "main.lua" to your local directory where psiblast profile features are generated in or have been generated in   
--------------------------------------------------------------------------------------------------

---- RUN TO TEST

lua main.lua -h

Usage: main.lua [options] <task><testf><umodel><outdir><labelf><psiExistFlag>

Goal: to predict AA labels for certain task on input fasta file

Arguments:
  <task>         task: cb513ssp, dssalltaskdssp, dssalltaskssp, dssalltasksar, dssalltasksaa, coilcoil, dna-binding, ppi, sp, tm
  <testf>        test fasta file to predict on
  <umodel>       a previously trained network model associated to your aimed task 
  <outdir>       subdirectory to save the predictions in
  <labelf>       test fasta file to evaluate accuracy (if not exists, just put <testf> here
  <psiExistFlag> psiExistFlag: [ 1 indicating psiBlast features exist, with system-defined dir  ; 
                                 2 indicating psiBlast features exist, with user-defined dir in psipath; 
                                 0 indictating psiBlast features do not exist ]


--------------------------------------------------------------------------------------------------
==> You can try the following sample commands 
----
---- lua main.lua -h 
----
---- lua main.lua cb513ssp ./test/fasta/input.cb513ssp.seq.faa ./models4shareOnline/cb513ssp.model.asc ./test/ ./test/fasta/input.cb513ssp.lab.faa 1 > ./test/cb513test.log &
----
---- lua main.lua dssalltaskdssp ./test/fasta/input.dssp4task.seq.faa ./models4shareOnline/dssalltaskdssp.model.asc ./test/ ./test/fasta/input.dssp.lab.faa 0 > ./test/dssalltaskdssp.log &
----
---- lua main.lua dssalltaskssp ./test/fasta/input.dssp4task.seq.faa ./models4shareOnline/dssalltaskssp.model.asc  ./test/ ./test/fasta/input.ssp.lab.faa  1 > ./test/dssalltaskssp.log
----
---- lua main.lua dssalltasksaa ./test/fasta/input.dssp4task.seq.faa ./models4shareOnline/dssalltasksaa.model.asc  ./test/ ./test/fasta/input.saa.lab.faa  1 > ./test/dssalltasksaa.log
----
---- lua main.lua dssalltasksar ./test/fasta/input.dssp4task.seq.faa ./models4shareOnline/dssalltasksar.model.asc  ./test/ ./test/fasta/input.sar.lab.faa  1 > ./test/dssalltasksar.log

==>  similar commands are applied to other tasks (coilcoil, dna-binding, sp, tm)

--------------------------------------------------------------------------------------------------
  
==> For PPI task, we use it non-viterbi model since it performs better 

---- lua main-nvit-ppi.lua ppi ./test/fasta/input.tppi.seq.faa ./models4shareOnline/TPPI-nonVit/tppi.psionly.model.asc.noviterbi ./test/ ./test/fasta/input.tppi.lab.faa 1 > ./test/log/ppi.log

--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------

