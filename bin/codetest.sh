cd ..
make clean all
cd bin
./drug_sim -input_deck input_deck_example.txt -hill_file drug/testing/few_IC50_verapamil.csv -herg_file herg/few_verapamil_2000.csv -init_file init_IC50_verapamil/few_state_only.csv
