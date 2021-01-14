# Code used to process data

* `get_parts_in_csv.py` convierte las partes descargadas en un csv único y sin repeticiones. Este archivo es `reviews_all.csv`.

* `process_csv.py` recolecta de `reviews_all.csv` los reviews más importantes, redefine las categorías y limita por producto, categoría y rate. De todo esto, salen los archivos `reviews_esp_full.csv` para español y `reviews_por_full.csv` para portugués. 

* `clean_dataset.py` utiliza los archivos `reviews_esp_full.csv` y `reviews_por_full.csv` para eliminar los reviews que son basura. Este genera un par de archivos intermedios que sirven para identificar los reviews basura, obteniendo como resultado final los archivos `reviews_esp_cleaned.csv` y `reviews_por_cleaned.csv`