# Paper12-PlosOne-DeepProteinMultitaskTagging

Y. Qi, M. Osh, J. Weston, W. Noble (2012) 
A unified multitask architecture for predicting local protein properties, PLoS ONE (March 2012) (URL), 

<BR>

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

