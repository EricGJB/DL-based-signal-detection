#!/usr/bin/env python
from transmitters import transmitters
from source_alphabet import source_alphabet
import analyze_stats
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft, cPickle, gzip
import random

dataset = {}
# The output format looks like this
# {('mod type', SNR): np.array(nvecs_per_key, 2, vec_length), etc}

nvecs_per_key = 1000
vec_length_list = [64,128,256,512,1024]
snr_vals = range(-20,20,1)
for vec_length in vec_length_list:
	for snr in snr_vals:
		print("snr is ", snr)
		for alphabet_type in transmitters.keys():
			for i,mod_type in enumerate(transmitters[alphabet_type]):
			  dataset[(mod_type.modname, snr)] = np.zeros([nvecs_per_key, 2, vec_length], dtype=np.float32)
			  # more vectors!
			  insufficient_modsnr_vectors = True
			  modvec_indx = 0
			  while insufficient_modsnr_vectors:
				  tx_len = int(10e3)
				  if mod_type.modname == "QAM16":
					  tx_len = int(20e3)
				  if mod_type.modname == "QAM64":
					  tx_len = int(30e3)
				  src = source_alphabet(alphabet_type, tx_len, True)
				  mod = mod_type()
				  snk = blocks.vector_sink_c()
				  tb = gr.top_block()
				  # connect blocks
				  tb.connect(src, mod, snk)
				  tb.run()
				  raw_output_vector = np.array(snk.data(), dtype=np.complex64)
				  # start the sampler some random time after channel model transients (arbitrary values here)
				  sampler_indx = random.randint(50, 500)
				  while sampler_indx + vec_length < len(raw_output_vector) and modvec_indx < nvecs_per_key:
					  sampled_vector = raw_output_vector[sampler_indx:sampler_indx+vec_length]
					  H = np.random.randn(1)+1j*np.random.randn(1)
					  sampled_vector = sampled_vector*H    
					  random_noise = np.random.randn(vec_length)+1j*np.random.randn(vec_length)
					  random_noise_energy = np.sum((np.abs(random_noise)**2))
					  signal_energy_expected = random_noise_energy*(10**(snr/10.0))
					  signal_energy = np.sum((np.abs(sampled_vector)**2))
					  sampled_vector = sampled_vector*(signal_energy_expected**0.5)/(signal_energy**0.5)
					  total_vector = sampled_vector + random_noise #AWGN
					  total_energy = np.sum(np.abs(total_vector)**2)
					  # energy normalization
					  total_vector = total_vector/(total_energy**0.5)
					  dataset[(mod_type.modname, snr)][modvec_indx,0,:] = np.real(sampled_vector)
					  dataset[(mod_type.modname, snr)][modvec_indx,1,:] = np.imag(sampled_vector)
					  # bound the upper end very high so it's likely we get multiple passes 
					  # through independent channels
					  sampler_indx += random.randint(vec_length, round(len(raw_output_vector)*.05))
					  modvec_indx += 1

				  if modvec_indx == nvecs_per_key:
					  # we're all done
					  insufficient_modsnr_vectors = False

	print("all done. writing to disk")
	cPickle.dump(dataset, file("./pkl_data/%d.pkl"%vec_length, "wb" ) )
