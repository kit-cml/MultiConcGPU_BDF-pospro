cd ..
make clean all
cd bin
# ./drug_sim -input_deck input_deck_example.txt -hill_file ./fresh_quinidine_cmax1_2_CIPA.csv -herg_file herg/quinidine_4000.csv -init_file ./fresh_1_2.csv
# ./drug_sim -input_deck input_deck_example.txt -hill_file ./euler_quinidine_cmax3_4_CIPA.csv -herg_file herg/quinidine_4000.csv -init_file ./quinidine_cmax3_4_1000pace_state_only.csv
# ./drug_sim -input_deck input_deck_example.txt -hill_file ./drug/testing2/IC50_quinidine_CIPA.csv -herg_file herg/quinidine_8000.csv -init_file ./init_euler_quinidine_1000pace/_state_only.csv
# ./drug_sim -input_deck input_deck_example.txt -hill_file drug/euler_optimal/IC50_optimal.csv -herg_file herg/optimal_quinidine.csv -init_file ./optimal_quinidine_cmax0.csv

#  ./drug_sim -input_deck input_deck_example.txt -hill_file ./drug_test/IC50_bepridil_1_2.csv -init_file ./drug/new_bepridil/1_2.csv  -herg_file ./herg/bepridil_4000.csv 
#  ./drug_sim -input_deck input_deck_example.txt -hill_file ./drug_test/IC50_bepridil_3_4.csv -init_file ./drug/new_bepridil/3_4.csv  -herg_file ./herg/bepridil_4000.csv 

# ./drug_sim -input_deck input_deck_example.txt -hill_file ./drug_test/IC50_quinidine.csv -init_file init_quinidine_cvar.csv  -herg_file ./herg/quinidine_2000.csv  -cvar_file cvar_optimised_2k.csv

./drug_sim -input_deck input_deck_example.txt -hill_file ./drug_test/IC50_bepridil_simple_1_2.csv -init_file init_bepridil_cmax1_2_cvar.csv  -herg_file ./herg/bepridil_4000.csv  -cvar_file optimized_4k.csv
./drug_sim -input_deck input_deck_example.txt -hill_file ./drug_test/IC50_bepridil_simple_3_4.csv -init_file init_bepridil_cmax3_4_cvar.csv  -herg_file ./herg/bepridil_4000.csv  -cvar_file optimized_4k.csv
