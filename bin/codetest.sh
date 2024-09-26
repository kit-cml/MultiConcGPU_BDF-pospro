cd ..
make clean all
cd bin
./drug_sim -input_deck input_deck_example.txt -hill_file drug/testing/ -herg_file herg/verapamil_2000.csv -init_file init_IC50_verapamil/_state_only.csv
