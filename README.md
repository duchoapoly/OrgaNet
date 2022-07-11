# OrgaNet

This project develops OrgaNet: A Robust Network for Subcellular Organelles Classification in Fluorescence Microscopy Images.
This works was published in the proceedings of 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC).

Duc Hoa Tran, Michel Meunier, Farida Cheriet, OrgaNet: A Robust Network for Subcellular Organelles Classification in Fluorescence Microscopy Images. EMBC 2020: 1887-1890, IEEE.

To train the network, using this command:
python train_organet.py --batch_size 16 --gamma 0.001 --gamma2 0.1 --ratio 8 >results/training_gamma0001_gamma2_01_ratio80___1.txt 2>&1
