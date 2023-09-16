******************************
* MuST-SHE EVALUATION SCRIPT *
*                            *
*  Fondazione Bruno Kessler  *
******************************

The evaluation script "mustshe_acc_v1.1.py" computes term coverage and gender accuracy scores 
for systems run on the MuST-SHE benchmark.

To work correctly, the script requires Python 3.

The script requires two mandatory arguments:

--input FILE 	the output of the system you want to evaluate
				Note that the output must be tokenized (eg. with Moses' tokenizer.perl)
				
--tsv-definition FILE	one of the MuST-SHE MONOLINGUAL tsv files (the Gold Standard) 
						contained in the directory "MuST-SHE-v1.1-data"

You can run "mustshe_acc.py --help" to get a list of the parameters taken by the script.


The script computes both overall scores and divided by MuST-SHE categories.

If you use this evaluation script, please cite:

Marco Gaido, Beatrice Savoldi, Luisa Bentivogli, Matteo Negri and Marco Turchi.
“Breeding Gender-Aware Direct Speech Translation Systems“
In Proceedings of the  28th International Conference on Computational Linguistics (COLING’2020), 
December 8-13 2020, Online, pp 3951-3964.


Contacts:
- Marco Gaido [mgaido@fbk.eu]
- Marco Turchi [turchi@fbk.eu]
- Luisa Bentivogli [bentivo@fbk.eu]