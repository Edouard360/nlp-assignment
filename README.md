To run the HMM algorithm from the root of the project, be careful 

`PYTHONPATH=$(pwd) python hmm_pcfg_files/hmm/hmm_nltk.py`

Please run:

`./hmm_pcfg_files/eval.py --reference_postags_filename=hmm_pcfg_files/dev_postags --candidate_postags_filename=hmm_pcfg_files/postags/postags_nltk`

To see the performances of the algorithm