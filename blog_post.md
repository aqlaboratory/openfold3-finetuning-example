# Tutorial: Finetuning OpenFold3 on data from OpenBind’s first data drop

When using a cofolding model in a drug discovery campaign, ideally the model should reflect the most up-to-date structural information available for the target. New structures may have recently been deposited in the PDB, or new experimental structures may be generated as the campaign progresses. 

Can a cofolding model be fine-tuned on new structural data and improve its performance on related structures? We recently explored this question using the first [OpenBind data release](https://www.biorxiv.org/content/10.64898/2026.08.27.747600v1), and found that fine-tuning OpenFold3 on fragment-bound structures improved its predictive performance on related follow-on compounds.

In this post, we will walk through what that workflow looks like in practice. We will process the data into the correct format for OpenFold3 training data, and show how to fine-tune the model on a desktop computer, a [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/). 

## 1: Dataset preparation

The OpenBind EV-A71 2A data release is available on [Zenodo](https://zenodo.org/records/22142262) as a single ~125 MB zip file (~600 MB unzipped). 

The dataset can then be processed using a single command: 

```
python runners/prepare_zenodo_data.py --input_dir dataset/OpenBind_EV-A71_2A --output_dir {folder path}
```

Alternatively, you can download the preprocessed dataset. 

More typically, you may want to identify relevant structures in the PDB. For this, you can use MMseqs2 (which has GPU accelerated variants) to search a sequence database constructed from PDB entries. In this case, we search the sequence for the Enterovirus A71 (EV-A71) / Coxsackievirus A16 (CVA16) 2A protease. Some, but not all of the structures from the OpenBind data release have already been deposited in the PDB. Searching this sequence, there are 673 hits with 100% sequence identity. From this folder of .cif files, you can prepare a dataset. 

```
python runners/prepare_pdb_data.py --input_dir {folder of .cif files} --output_dir {folder path}
```

## 2: Creating your train/validation caches 

Once your dataset is prepared, you can choose what subset to use as training, and what portion to use as validation. Assuming you have a list of ids for train and validation, you can run the following:

```
python runners/create_dataset_cache.py train --ids {train_ids.txt} --dataset_dir {folder path}
python runners/create_dataset_cache.py validation --ids {val_ids.txt} --dataset_dir {folder path}
```

This creates files which will be used in the next step to specify to the model what the training and validation set is. 

## 3: Preparing training set yaml 

Now that the dataset has been prepared, you have to prepare a training yaml which specifies the newly generated training and validation datasets. If you’re doing structures processed from the PDB, can just use the CCD file provided by the PDB. 

```
alignments_directory: none
alignment_db_directory: none
alignment_array_directory: {dataset_dir}/alignment_arrays
dataset_cache_file: {dataset_dir}/caches/train_cache.json
target_structures_directory: {dataset_dir}/preprocessed/structure_files
target_structure_file_format: npz
reference_molecule_directory: {dataset_dir}/preprocessed/reference_mols
template_cache_directory: none
template_structure_array_directory: {placeholder directory}
template_structures_directory: none
template_file_format: npz
ccd_file: {dataset_dir}/merged_cifs/reference_data/components_harmonized.cif
```

#### 

## 4: Start training\!

With your completed training cache, you can now launch your training run. 

```
srun python -m openfold3.run_openfold train \   
      --runner_yaml "${RUNNER_YAML}" \
      --seed "${SEED}"
```

Here we fine-tune OpenFold3-p2 on the 79 fragment-bound structures for 1,000 steps using a batch size of 8, a learning rate of 10^-3, and a 100-step linear warm-up. This takes approximately eight hours on eight NVIDIA GH200 GPUs, or 50 hours on a single desktop GPU on a DGX-Spark. Depending on the target or the amount of data, it may take more or less time. 

![][image1]

After training, we have a new OpenFold3 checkpoint that can be run through the same inference pipeline as the base model.

## Evaluating structural accuracy

Was all this effort worth it? Did the fine-tuning improve the model’s ability to predict follow-on compounds? We find that for this particular drug target, fine-tuning does dramatically improve the predictive capability. 

![][image2]  
Figure from OpenBind data release [paper](https://www.biorxiv.org/content/10.64898/2026.08.27.747600v1). 

## What this means in practice

For this target, a small set of fragment-bound structures could improve pose prediction and virtual-screening discrimination on related follow-on compounds. And fine-tuning could even be done on a single desktop GPU.

In practice, this means a cofolding model does not have to remain fixed during a campaign. As new structures are generated, they can be trained on and used to improve the next round of predictions.

We are still understanding the conditions under which fine tuning is helpful. We expect that it may help for certain targets more than others, and for certain test entries more than others. But we hope the method described here will help others get started on fine tuning on their own dataset. 
