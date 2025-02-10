To reproduce the results:
python motif_explainer.py
python bashapes_explainer.py
python ba2motif_explainer.py

If you want to retrain and fine tune the model, you need to modify BA2motif_generater.py, motif_generater.py, model.py for each dataset.
For ba2motif dataset:
python preprocess_ba2motif.py
python ba2motif_att.py
python ba2motif_explainer.py

For bashape dataset:
python preprocess_bashapes.py
python bashape_motif.py
python bashapes_explainer.py

For mutagenicity dataset: (please uncomment training part)
python motif_explainer.py
