"""
Author: Alicia Olivares-Gil

This script performs the search for homologous sequences in the UniRef100 database using jackhmmer. 
Then post-process the MSA generated. 
Finally uses plmc software to infer DCA statistical model. 

Syntaxis: jackhmmer.py [UniRef100 .fasta path]

Example: jackhmmer.py ../uniref100.fasta

Folder "raw_data" in root must contain a sub-folder for each dataset considered. Each sub-folder must contain a .fasta with the wild type sequence and will contain all the resulting files for that specific dataset after the search and MSA post-processing. 

Before running the script: 

raw_data
├──avgfp
│   └──avgfp.fasta
├──bg_strsq
│   └──bg_strsq.fasta
└──...


After running the script: 

raw_data
├──avgfp
│   └──avgfp.fasta
│   └──avgfp_jhmmer.sto
│   └──avgfp_jhmmer.a2m
│   └──avgfp_plmc.params
├──bg_strsq
│   └──bg_strsq.fasta
│   └──...
└──...

""" 
import os
import sys
from Bio import SeqIO
from sto2a2m import convert_sto2a2m

def run_jackhmmer(uniref_path, data_dict): 
    """
    This script performs the search for homologous sequences in the UniRef100 database using jackhmmer. 
    Then post-process the MSA generated (positions with more than 30% gaps and sequences with more than 
    50% gaps are excluded). Finally uses plmc software to infer DCA statistical model.  
    
    :param uniref_path: UniRef100 .fasta path
    :param data_dict: Dictionary generated in "main" containing necessary file names. 
    
    :returns None (Generates .sto, .a2m and .params files for each dataset.)
    
    """
    for key in data_dict.keys(): 
        fasta = data_dict[key]['fasta']
        sto = data_dict[key]['sto']
        a2m = data_dict[key]['a2m']
        params = data_dict[key]['params']

        wt = SeqIO.read(open(fasta), 'fasta')
        length = len(wt)
        incT = length*0.5
        
        #CALL JACKHMMER
        with open('jackhmmer_log.log', 'a') as writer:
            writer.write('>>> '+key+':\n')
            writer.write('Running jackhmmer\n')
        os.system('jackhmmer --incT '+str(incT)+' --cpu 64 --noali -A '+sto+' '+fasta+' '+uniref_path) 
        
        #CONVERT .sto TO .a2m
        with open('jackhmmer_log.log', 'a') as writer:
            writer.write('Converting .sto to .a2m\n')
        n_seqs,n_active_sites,n_sites=convert_sto2a2m(sto, 0.3, 0.5)
        le = 0.2*(n_active_sites-1)
        
        #CALL PLMC
        with open('jackhmmer_log.log', 'a') as writer:
            writer.write('Running plmc\n')
            writer.write('\n')
        os.system('../../../plmc/bin/plmc -o '+params+' -n 64 -le '+str(le)+' -m 3500 -g -f '+key+' '+a2m)
        

if __name__=="__main__":
    
    data_path = os.path.join('..', 'raw_data')
    
    if len(sys.argv) == 1:
        uniref_path = sys.argv[1]
    else: 
        raise SyntaxError('''UniRef100 .fasta path must be specified a parameter. 
                             \n\tSyntaxis: jackhamer.py [UniRef100 .fasta path]
                             \n\tExample: jackhmmer.py ../uniref100.fasta''')
        
    data_dict = dict()

    #Comment and Decomment to select datasets on which to perform jackhmmer search. 

    data_dict['avgfp'] = {'fasta': os.path.join(data_path, 'avgfp', 'avgfp.fasta'), 
                          'sto': os.path.join(data_path, 'avgfp', 'avgfp_jhmmer.sto'), 
                          'a2m': os.path.join(data_path, 'avgfp', 'avgfp_jhmmer.a2m'), 
                          'params': os.path.join(data_path, 'avgfp', 'avgfp_plmc.params')} 
    
    data_dict['bg_strsq'] = {'fasta': os.path.join(data_path, 'bg_strsq', 'bg_strsq.fasta'), 
                             'sto': os.path.join(data_path, 'bg_strsq', 'bg_strsq_jhmmer.sto'), 
                             'a2m': os.path.join(data_path, 'bg_strsq', 'bg_strsq_jhmmer.a2m'),
                             'params': os.path.join(data_path, 'bg_strsq', 'bg_strsq_plmc.params')}
    
    data_dict['blat_ecolx'] = {'fasta': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx.fasta'), 
                               'sto': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_jhmmer.sto'), 
                               'a2m': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_jhmmer.a2m'), 
                               'params': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_plmc.params')}
    
    data_dict['brca1_human'] = {'fasta': os.path.join(data_path, 'brca1_human', 'brca1.fasta'), 
                                'sto': os.path.join(data_path, 'brca1_human', 'brca1_human_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'brca1_human', 'brca1_human_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'brca1_human', 'brca1_human_plmc.params')}
    
    data_dict['gal4_yeast'] = {'fasta': os.path.join(data_path, 'gal4_yeast', 'gal4_yeast.fasta'), 
                               'sto': os.path.join(data_path, 'gal4_yeast', 'gal4_yeast_jhmmer.sto'), 
                               'a2m': os.path.join(data_path, 'gal4_yeast', 'gal4_yeast_jhmmer.a2m'), 
                               'params': os.path.join(data_path, 'gal4_yeast', 'gal4_yeast_plmc.params')}
    
    data_dict['hg_flu'] = {'fasta': os.path.join(data_path, 'hg_flu', 'hg_flu.fasta'), 
                           'sto': os.path.join(data_path, 'hg_flu', 'hg_flu_jhmmer.sto'), 
                           'a2m': os.path.join(data_path, 'hg_flu', 'hg_flu_jhmmer.a2m'), 
                           'params': os.path.join(data_path, 'hg_flu', 'hg_flu_plmc.params')}
    
    data_dict['hsp82_yeast'] = {'fasta': os.path.join(data_path, 'hsp82_yeast', 'hsp82_yeast.fasta'), 
                                'sto': os.path.join(data_path, 'hsp82_yeast', 'hsp82_yeast_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'hsp82_yeast', 'hsp82_yeast_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'hsp82_yeast', 'hsp82_yeast_plmc.params')}
    """
    
    data_dict['kka2_klepn'] = {'fasta': os.path.join(data_path, 'kka2_klepn', 'kka2_klepn.fasta'), 
                               'sto': os.path.join(data_path, 'kka2_klepn', 'kka2_klepn_jhmmer.sto'), 
                               'a2m': os.path.join(data_path, 'kka2_klepnt', 'kka2_klepnt_jhmmer.a2m'), 
                               'params': os.path.join(data_path, 'kka2_klepnt', 'kka2_klepnt_plmc.params')}
    
    """
    data_dict['mth3_haeaestabilized'] = {'fasta': os.path.join(data_path, 'mth3_haeaestabilized', 'mth3.fasta'), 
                                         'sto': os.path.join(data_path, 'mth3_haeaestabilized', 'mth3_haeaestabilized_jhmmer.sto'), 
                                         'a2m': os.path.join(data_path, 'mth3_haeaestabilized', 'mth3_haeaestabilized_jhmmer.a2m'), 
                                         'params': os.path.join(data_path, 'mth3_haeaestabilized', 'mth3_haeaestabilized_plmc.params')}
                                        
    data_dict['pabp_yeast'] = {'fasta': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast.fasta'), 
                                'sto': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast_plmc.params')}
    
    data_dict['polg_hcvjf'] = {'fasta': os.path.join(data_path, 'polg_hcvjf', 'polg_hcvjf.fasta'), 
                                'sto': os.path.join(data_path, 'polg_hcvjf', 'polg_hcvjf_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'polg_hcvjf', 'polg_hcvjf_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'polg_hcvjf', 'polg_hcvjf_plmc.params')}
    
    data_dict['rl401_yeast'] = {'fasta': os.path.join(data_path, 'rl401_yeast', 'rl401.fasta'), 
                                'sto': os.path.join(data_path, 'rl401_yeast', 'rl401_yeast_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'rl401_yeast', 'rl401_yeast_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'rl401_yeast', 'rl401_yeast_plmc.params')}
    
    data_dict['ube4b_mouse'] = {'fasta': os.path.join(data_path, 'ube4b_mouse', 'ueb_mouse.fasta'), 
                                'sto': os.path.join(data_path, 'ube4b_mouse', 'ube4b_mouse_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'ube4b_mouse', 'ube4b_mouse_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'ube4b_mouse', 'ube4b_mouse_plmc.params')}
    
    data_dict['yap1_human'] = {'fasta': os.path.join(data_path, 'yap1_human', 'yap1_human.fasta'), 
                                'sto': os.path.join(data_path, 'yap1_human', 'yap1_human_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'yap1_human', 'yap1_human_jhmmer.a2m'),
                                'params': os.path.join(data_path, 'yap1_human', 'yap1_human_plmc.params')}
    
    #Run jackhmmer searcha and post-processing MSA. 
    run_jackhmmer(uniref_path, data_dict)