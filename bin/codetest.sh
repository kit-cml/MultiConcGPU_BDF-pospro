cd ..
make clean all
cd bin
./drug_sim -input_deck input_deck_example.txt -hill_file ./euler_quinidine_cmax3_4_CIPA.csv -herg_file herg/quinidine_4000.csv -init_file ./quinidine_cmax3_4_1000pace_state_only.csv
