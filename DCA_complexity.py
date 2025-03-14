import os
import sys
from Bio import SeqIO
from scripts.sto2a2m import convert_sto2a2m
from scripts.utils import read_variants, read_a2m, generate_unambiguous_homologs
from scripts.encoding import SequencesDCAEncoder

import time
import tracemalloc 

if __name__=="__main__": 
    
    
    data_path = os.path.join('raw_data')
    uniref_path = os.path.join('uniref100.fasta')
    fasta = os.path.join(data_path, 'avgfp', 'avgfp.fasta')
    sto = os.path.join(data_path, 'avgfp', 'avgfp_jhmmer_v2.sto')
    a2m = os.path.join(data_path, 'avgfp', 'avgfp_jhmmer_v2.a2m')
    params = os.path.join(data_path, 'avgfp', 'avgfp_plmc_v2.params')
    wt = SeqIO.read(open(fasta), 'fasta')
    length = len(wt)
    incT = length*0.5
    
    print("CALL JACKHMMER")
    t_start = time.time()
    tracemalloc.start()
    
    #CALL JACKHMMER
    os.system('jackhmmer --incT '+str(incT)+' --cpu 64 --noali -A '+sto+' '+fasta+' '+uniref_path) 
    
    t_end = time.time()
    
    with open('jackhmmer_log.log', 'a') as writer:
        writer.write(f'time: {t_end-t_start}')
        writer.write(f'mem: {tracemalloc.get_traced_memory()}')
    tracemalloc.stop()
    
    print("CONVERT .sto TO .a2m AND CALL PLMC")
    t_aux = time.time()
    tracemalloc.start()
    
    #CONVERT .sto TO .a2m
    n_seqs,n_active_sites,n_sites=convert_sto2a2m(sto, 0.3, 0.5)
    le = 0.2*(n_active_sites-1)
    #CALL PLMC
    os.system('../../../plmc/bin/plmc -o '+params+' -n 64 -le '+str(le)+' -m 3500 -g -f avgfp '+a2m)
    
    t_end = time.time()
    
    with open('jackhmmer_log.log', 'a') as writer:
        writer.write(f'time: {t_end-t_aux}')
        writer.write(f'mem: {tracemalloc.get_traced_memory()}')
    tracemalloc.stop()
    
    #read variants
    start_pos = 1
    csv = os.path.join(data_path, 'avgfp', 'avgfp.csv')
    sequences, y, variants = read_variants(csv, fasta, start_pos=start_pos)
    
    print("ENCODE SEQUENCES")
    t_aux = time.time()
    tracemalloc.start()
    
    #encode sequences 
    dcae = SequencesDCAEncoder(wt, start_pos, params)
    Xl_dcae, y_dcae, indexes = dcae.encode_variants(sequences, y, variants)
    
    t_end = time.time()
    
    with open('jackhmmer_log.log', 'a') as writer:
        writer.write(f'time: {t_end-t_aux}')
        writer.write(f'mem: {tracemalloc.get_traced_memory()}')
    tracemalloc.stop()
    
    with open('jackhmmer_log.log', 'a') as writer:
        writer.write("TOTAL TIME")
        writer.write(f'time: {t_end-t_start}')
