cd ..
make clean all
cd bin
./drug_sim -input_deck input_deck_example.txt -hill_file ./fresh_quinidine_cmax1_2_CIPA.csv -herg_file herg/quinidine_4000.csv -init_file ./fresh_1_2.csv
# ./drug_sim -input_deck input_deck_example.txt -hill_file ./euler_quinidine_cmax3_4_CIPA.csv -herg_file herg/quinidine_4000.csv -init_file ./quinidine_cmax3_4_1000pace_state_only.csv
# ./drug_sim -input_deck input_deck_example.txt -hill_file ./drug/testing2/IC50_quinidine_CIPA.csv -herg_file herg/quinidine_8000.csv -init_file ./init_euler_quinidine_1000pace/_state_only.csv
# ./drug_sim -input_deck input_deck_example.txt -hill_file drug/euler_optimal/IC50_optimal.csv -herg_file herg/optimal_quinidine.csv -init_file ./optimal_quinidine_cmax0.csv
