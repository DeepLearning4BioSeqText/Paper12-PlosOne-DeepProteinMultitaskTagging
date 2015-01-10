import os, sys

inFile=sys.argv[1]
output_directory=sys.argv[2]

sys.stderr.write("Input fasta file:"+inFile+"\n")
sys.stderr.write("Output directory:"+output_directory+"\n")

#please replace the following blast position 
#binDir = "/net/noble/vol1/home/noble/bin"
binDir = "/net/multiSequence/blast/blast-2.2.21/bin"
blastbin=os.path.expanduser(binDir + "/blastpgp")
makematbin=os.path.expanduser(binDir + "/makemat")
# please replace the following db position
nrdb = "/net/multiSequence/blast/db/nrfilt"

IN=open(inFile, "r")
fasta_hash={}
for line in IN:
    if line[0]=='>':
        name=line[1:].strip()
        fasta_hash[name]=""
    else:
        fasta_hash[name]+=line
IN.close()

for name in fasta_hash:
    outname=name.split()[0].replace("|", "_")
    sys.stderr.write("Processing "+outname+"\n")
    outputpath = os.path.join(output_directory, name[0], name[1])+"/"
    if not os.path.isdir(outputpath):
        if not os.path.isdir(os.path.join(output_directory, name[0])+"/"):
            os.mkdir(os.path.join(output_directory, name[0])+"/")
        os.mkdir(outputpath)
    if not os.path.isfile(outputpath+outname+".mtx"):
        OUT=open("input.fasta", "w")
        OUT.write(">"+name+"\n")
        OUT.write(fasta_hash[name])
        OUT.close()
        os.system(blastbin+" -d " +nrdb +" -i input.fasta -e 10 -h 0.001 -j 3 -m 8 -Q "+outputpath+outname+".mtx -o "+outputpath+outname+".out")
