cd ..
make clean all
cd bin
./drug_sim -input_deck input_deck.txt -hill_file drug/testing/ -herg_file herg/verapamil_2000.csv -init_file control_me/init_drug_control.csv
